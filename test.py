import mlflow
import mlflow.pyfunc
import pandas as pd

# โหลดโมเดล (เปลี่ยน <RUN_ID> เป็น ID ของคุณจริงๆ)
model = mlflow.pyfunc.load_model("runs:/ea05584d9f864e0187e72b163f72cc55/model")

# สร้าง DataFrame จากข้อมูลตัวอย่าง
data = pd.DataFrame({
    "Area": [28395],
    "Perimeter": [610.291],
    "MajorAxisLength": [208.1781167],
    "MinorAxisLength": [173.888747],
    "AspectRation": [1.197191424],
    "Eccentricity": [0.5498121871],
    "ConvexArea": [28715],
    "EquivDiameter": [190.1410973],
    "Extent": [0.7639225182],
    "Solidity": [0.9888559986],
    "roundness": [0.9580271263],
    "Compactness": [0.9133577548],
    "ShapeFactor1": [0.007331506135],
    "ShapeFactor2": [0.003147289167],
    "ShapeFactor3": [0.8342223882],
    "ShapeFactor4": [0.998723889]
})

# ทำนาย
predictions = model.predict(data)
print("Prediction:", predictions)
