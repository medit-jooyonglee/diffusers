# my_project/
# │
# ├── models/
# │   ├── __init__.py
# │   └── bisenet.py       # GitHub에서 가져온 모델 정의 파일
# ├── weights/
# │   └── bisenet_celebA.pth
# └── main.py              # 실행 메인 코드

# 가중치를 저장할 weights 폴더 생성 및 다운로드
# mkdir -p weights
# curl -L -o weights/bisenet_celebA.pth https://github.com/yakhyo/face-parsing/releases/download/v0.1.0/bisenet_resnet18_celebamaskhq.pth


import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import glob


# SEE my/modules/face_parsing/README.md
# pip install gdown
# gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O weights/79999_iter.pth

from my.modules.face_parsing.model import BiSeNet

# 얼굴 검출기 (OpenCV 내장 YuNet)
# curl -L -o weights/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# zllrunning/face-parsing.PyTorch (CelebAMask-HQ) 19 클래스
CLASSES = [
    'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat',
    'teeth',  # BiSeNet 출력이 아니라 mouth 영역에서 후처리로 분리
]
MOUTH, U_LIP, L_LIP, TEETH = 11, 12, 13, 19

# 클래스별 색상 (BGR)
PALETTE = np.array([
    [0, 0, 0],        # background
    [128, 178, 255],  # skin
    [0, 128, 255],    # l_brow
    [0, 200, 255],    # r_brow
    [255, 128, 0],    # l_eye
    [255, 200, 0],    # r_eye
    [255, 0, 255],    # eye_g
    [128, 0, 255],    # l_ear
    [200, 0, 255],    # r_ear
    [255, 255, 0],    # ear_r
    [0, 255, 255],    # nose
    [0, 255, 0],      # mouth (입 안/치아)
    [0, 0, 255],      # u_lip
    [80, 0, 200],     # l_lip
    [128, 255, 128],  # neck
    [0, 128, 128],    # neck_l
    [180, 180, 180],  # cloth
    [40, 40, 160],    # hair
    [255, 128, 128],  # hat
    [255, 0, 128],    # teeth
], dtype=np.uint8)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_parser(weight_path, device):
    model = BiSeNet(n_classes=19)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def detect_faces(detector, img_bgr, score_thresh=0.6, max_side=640, pad_ratio=0.25):
    # YuNet은 큰 입력에서 느리고(OpenCV 5에서는 segfault도 남), 화면을 꽉 채운 얼굴은 놓침
    # -> 축소 후 검출, 실패하면 테두리 패딩해서 재시도하고 박스를 원본 좌표로 복원
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    detector.setScoreThreshold(score_thresh)
    for pad in (0, int(max(small.shape[:2]) * pad_ratio)):
        padded = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        detector.setInputSize((padded.shape[1], padded.shape[0]))
        _, faces = detector.detect(padded)
        if faces is not None:
            return [(f[:4] - np.array([pad, pad, 0, 0], dtype=f.dtype)) / scale for f in faces]
    return []


def face_crop_box(box, img_w, img_h, scale=2.0):
    # CelebAMask-HQ처럼 머리카락/목까지 들어오도록 얼굴 박스를 정사각형으로 확장
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2 - 0.1 * h
    side = int(max(w, h) * scale)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    return x0, y0, x0 + side, y0 + side


def crop_with_pad(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    return cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)


@torch.no_grad()
def parse_crop(model, crop_bgr, device, size=512):
    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    out = model(TRANSFORM(rgb).unsqueeze(0).to(device))
    if isinstance(out, (tuple, list)):
        out = out[0]
    # argmax 전에 확률맵을 크롭 해상도로 upsample해야 경계가 매끄러움
    prob = F.interpolate(out.softmax(1), size=(h, w), mode='bilinear', align_corners=False)
    return prob.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)


def split_teeth(img_bgr, parsing, min_pixels=20):
    # mouth(입 안) 중 밝고 채도 낮은 픽셀을 치아로 분리 (혀/구강은 어둡거나 붉음)
    mouth = parsing == MOUTH
    if mouth.sum() < min_pixels:
        return
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a = lab[..., 0][mouth], lab[..., 1][mouth]
    l_thresh, _ = cv2.threshold(l.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    teeth = (l > l_thresh) & (a < 150)
    ys, xs = np.nonzero(mouth)
    parsing[ys[teeth], xs[teeth]] = TEETH


def parse_image(model, detector, img_bgr, device):
    h, w = img_bgr.shape[:2]
    parsing = np.zeros((h, w), dtype=np.uint8)
    boxes = detect_faces(detector, img_bgr)
    if not boxes:
        # 얼굴 검출 실패 시 전체 이미지로 추론
        return parse_crop(model, img_bgr, device), 0

    for box in boxes:
        x0, y0, x1, y1 = face_crop_box(box, w, h)
        crop_parsing = parse_crop(model, crop_with_pad(img_bgr, x0, y0, x1, y1), device)
        # 패딩 영역은 버리고 원본 이미지 안쪽만 붙여넣기
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        sub = crop_parsing[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        region = parsing[cy0:cy1, cx0:cx1]
        region[sub > 0] = sub[sub > 0]
    split_teeth(img_bgr, parsing)
    return parsing, len(boxes)


def blend_parsing(img_bgr, parsing, alpha=0.5):
    color = PALETTE[parsing]
    fg = parsing > 0
    result = img_bgr.copy()
    result[fg] = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)[fg]
    # 클래스 경계선
    edges = cv2.Canny(parsing, 0, 0) > 0
    result[edges] = (0.4 * result[edges] + 0.6 * 255).astype(np.uint8)
    return result


def draw_legend(parsing, height):
    ids = [i for i in np.unique(parsing) if i > 0]
    legend = np.full((height, 160, 3), 255, dtype=np.uint8)
    for row, i in enumerate(ids):
        y = 10 + row * 22
        cv2.rectangle(legend, (8, y), (26, y + 16), PALETTE[i].tolist(), -1)
        cv2.putText(legend, CLASSES[i], (34, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return legend


def predict_face_parsing(input_dir, weight_path='weights/79999_iter.pth',
                         detector_path='weights/face_detection_yunet_2023mar.onnx', output_path='outputs/seg_masks'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    model = load_parser(weight_path, device)
    detector = cv2.FaceDetectorYN.create(detector_path, '', (320, 320))
    os.makedirs(output_path, exist_ok=True)

    for image_path in sorted(glob.glob(os.path.join(input_dir, '**/*.png'), recursive=True)):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            continue

        parsing, n_faces = parse_image(model, detector, img_bgr, device)
        result = blend_parsing(img_bgr, parsing)
        result = np.concatenate([img_bgr, result, draw_legend(parsing, img_bgr.shape[0])], axis=1)

        output_file = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_file, result)
        print(f"{os.path.basename(image_path)}: 얼굴 {n_faces}개, 입술 픽셀 {np.sum((parsing == U_LIP) | (parsing == L_LIP))}, "
              f"입 안 픽셀 {np.sum(parsing == MOUTH)}, 치아 픽셀 {np.sum(parsing == TEETH)}")


if __name__ == '__main__':
    predict_face_parsing(
        input_dir='my/samples/faces/samples', 
                         output_path='outputs/seg_masks2')
