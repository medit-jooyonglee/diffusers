docker build후 테스트까지 해보려고함 torch기반,
  docker/design-tune-server/Dockerfile 한번더 검토해주고. 
  aws s3용 / ec2?(local테스트는 완료 fp8동작 완료) docker파일도 하나 마들어보자. 비슷하게 가되 시스템을 아직 이해를 못해가 니가 기본 짜줘야한다. 상황은 s3버킷 내용 이름 받앗고(테스트용)( - 배포시에는 어떻게 해야되는지 모르겟음.(당근 버킷 이름 변경되겟지,.?) 우선 s3에 필요한 weights파일 올리거다. flux2 4b모델. 그걸 아마 my / 쪽 문서에서처럼. 그리고 최종 배포 yaml에 대한 configure지정. 소스코드로 git으로 올리면될것 같고 사전 임베딩 결과를 s3에 올려야겟지
  s3 : weights & 사전 임베딩

  mediapipe도 우선 올려야될것 같고, (사용안할수도잇지만)
  - 현재 local환경에서는 테스트안됨. 우선dockerfile만 잘 만들ㅇㅊ어줘봐

  - aot로 할 , not compille fp8 op. 할지는 좀더 면밀 테스트후 결정.

