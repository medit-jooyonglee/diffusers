import glob
import io
import os
from urllib.parse import unquote_plus

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def decode_caption(text: str) -> str:
    if not text:
        return ""
    return unquote_plus(text).strip()

"""_summary_

commoncatalog-cc-by dataset
nohup hf download common-canvas/commoncatalog-cc-by --repo-type dataset --include "*/least_dim_range=512-768/*.parquet" --local-dir /data1/jooyonglee/commoncatalog-cc-by > logs/commondata.log 2>&1 &
nohup hf download common-canvas/commoncatalog-cc-by --repo-type dataset --include "*/least_dim_range=768-1024/*.parquet" --local-dir /data1/jooyonglee/commoncatalog-cc-by > logs/com_768_1024.log 2>&1 &
"""

class CommonCatalogDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        image_key="jpg",
        caption_key="blip2_caption",
        fallback_caption_key="caption",
        batch_size=64,
        skip_broken_files=True,
        process_index=0,
        num_processes=1,
        max_samples=None,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.image_key = image_key
        self.caption_key = caption_key
        self.fallback_caption_key = fallback_caption_key
        self.batch_size = batch_size
        self.skip_broken_files = skip_broken_files
        self.process_index = process_index
        self.num_processes = num_processes
        self.max_samples = max_samples
        if not 0 <= process_index < num_processes:
            raise ValueError(
                f"process_index must be in [0, {num_processes}), got {process_index}"
            )

        self.files = sorted(
            glob.glob(
                os.path.join(root_dir, "**", "*.parquet"),
                recursive=True,
            )
        )

        if self.process_index == 0:
            print(f"[CommonCatalog] parquet files: {len(self.files)}")

    def _parse_image(self, image_data):
        if image_data is None:
            return None

        # HuggingFace Image struct 형태
        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes", None)
            image_path = image_data.get("path", None)

            if image_bytes is not None:
                return Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

            if image_path:
                return Image.open(image_path).convert("RGB")

            return None

        # raw bytes 형태
        if isinstance(image_data, (bytes, bytearray)):
            return Image.open(
                io.BytesIO(image_data)
            ).convert("RGB")

        return None

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        shard_index = self.process_index * num_workers + worker_id
        num_shards = self.num_processes * num_workers
        files = self.files[shard_index::num_shards]
        shard_limit = None
        if self.max_samples is not None:
            process_limit = self.max_samples // self.num_processes
            shard_limit = process_limit // num_workers
            shard_limit += int(worker_id < process_limit % num_workers)
        yielded = 0
        if shard_limit == 0:
            return

        for parquet_path in files:

            try:
                pf = pq.ParquetFile(parquet_path)

                available_columns = set(pf.schema.names)

                columns = [self.image_key]

                if self.caption_key in available_columns:
                    columns.append(self.caption_key)

                if self.fallback_caption_key in available_columns:
                    columns.append(self.fallback_caption_key)

                for batch in pf.iter_batches(
                    batch_size=self.batch_size,
                    columns=columns,
                ):
                    data = batch.to_pydict()

                    num_rows = len(data[self.image_key])

                    for i in range(num_rows):
                        try:
                            image = self._parse_image(
                                data[self.image_key][i]
                            )

                            if image is None:
                                continue

                            caption = ""

                            # 1순위: BLIP2 caption
                            if self.caption_key in data:
                                caption = data[self.caption_key][i] or ""

                            # fallback: original caption
                            if not caption and self.fallback_caption_key in data:
                                caption = decode_caption(
                                    data[self.fallback_caption_key][i]
                                )

                            if not caption:
                                continue

                            yield {
                                "image": image,
                                "caption": caption,
                                "source_file": parquet_path,
                            }
                            yielded += 1
                            if shard_limit is not None and yielded >= shard_limit:
                                return

                        except Exception as e:
                            print(
                                f"[WARN] sample skip: "
                                f"{parquet_path} row={i} err={e}"
                            )
                            continue

            except Exception as e:
                print(
                    f"[ERROR] parquet skip: {parquet_path}\n"
                    f"        {type(e).__name__}: {e}"
                )

                if not self.skip_broken_files:
                    raise

                continue


def main_dataload():
    root = "/data1/jooyonglee/commoncatalog-cc-by/0/least_dim_range=512-768/"

    dataset = CommonCatalogDataset(
        root_dir=root,
        image_key="jpg",
        caption_key="blip2_caption",
        fallback_caption_key="caption",
        batch_size=64,
        skip_broken_files=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,  # 먼저 0으로 검증
    )



    for idx, sample in enumerate(loader):
        image = sample["image"]
        caption = sample["caption"]

        print(
            f"[{idx}] "
            f"size={image.size} "
            f"caption={caption[:120]}"
        )

        # if idx == 0:
        image.save(f"outputs/test_commoncatalog{idx}.jpg")

        if idx >= 20:
            break
        
        
def prompt_saving(
    base_root='',
    num_save=200,
    save_path='my/samples/commoncatalog'
):
    
    base_root = base_root or "/data1/jooyonglee/commoncatalog-cc-by/0/least_dim_range=512-768/"

    dataset = CommonCatalogDataset(
        root_dir=base_root,
        image_key="jpg",
        caption_key="blip2_caption",
        fallback_caption_key="caption",
        batch_size=64,
        skip_broken_files=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,  # 먼저 0으로 검증
    )



    prompts = []
    num_sample = 500
    for idx, sample in enumerate(loader):
        if idx >= num_sample:
            break
        image = sample["image"]
        caption = sample["caption"]

        # print(
        #     f"[{idx}] "
        #     f"size={image.size} "
        #     f"caption={caption[:120]}"
        # )
        prompts.append(caption)

    
    
    save_filename = os.path.join(save_path, "prompts.txt")
    os.makedirs(save_path, exist_ok=True)
    with open(save_filename, 'w', encoding='utf-8') as f:
        f.writelines(line + "\n" for line in prompts)
    print(f"Prompts saved to {save_filename}")
        
        # if idx == 0:
        # image.save(f"outputs/test_commoncatalog{idx}.jpg")

        # if idx >= 20:
            # break
        
    
if __name__ == "__main__":
    prompt_saving()
    # main_dataload()
