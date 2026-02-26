from ultralytics import YOLO

model=YOLO('models/last.pt')

results=model.predict('input_videos\A1606b0e6_0 (10).mp4',save=True)
print(results[0])
print('==================================')
for box in results[0].boxes:
    print(box)