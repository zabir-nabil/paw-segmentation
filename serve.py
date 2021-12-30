from fastapi import Request, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import base64
import uuid
import uvicorn
import numpy as np
import cv2
import torch

app = FastAPI()

# handling cors
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.model = torch.load('../best_model.pth', map_location=torch.device('cpu'))

def preprocess_input_smp(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if input_space == "BGR":
        x = x[..., ::-1].copy()
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0
    if mean is not None:
        mean = np.array(mean)
        x = x - mean
    if std is not None:
        std = np.array(std)
        x = x / std
    return x

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

@app.get("/")
async def index():
    return {"msg": "server is running."}

@app.post("/upload")
async def post_base64Image(request: Request):
    imgstr = await request.json()

    try:
        imgdata = base64.b64decode(imgstr.get("base64_image"))
        filename = f"dumped/img_{uuid.uuid4().hex}.png"
        with open(filename, 'wb') as f:
            f.write(imgdata)

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))

        image = preprocess_input_smp(image, input_space='RGB', input_range=[0, 1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = to_tensor(image)

        x_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)
        pr_mask = app.model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_mask = (pr_mask * 255.0).astype(np.uint8)

        if imgstr.get("mask_type") == "rgba":
            rgb = cv2.merge((pr_mask,pr_mask,pr_mask))
            rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)

            rgba[:, :, 3] = pr_mask

            _, encoded_img = cv2.imencode('.PNG', rgba)
            encoded_img = base64.b64encode(encoded_img)
            return {'mask': encoded_img}
        elif imgstr.get("mask_type") == "single_channel":
            _, encoded_img = cv2.imencode('.PNG', pr_mask)
            encoded_img = base64.b64encode(encoded_img)
            return {'mask': encoded_img}
        else:
            rgb = cv2.merge((pr_mask,pr_mask,pr_mask))
            _, encoded_img = cv2.imencode('.PNG', rgb)
            encoded_img = base64.b64encode(encoded_img)
            return {'mask': encoded_img}


    except:
        return {'msg': 'mask generation failed'}

if __name__ == '__main__':
    uvicorn.run("serve:app", port=80, host='0.0.0.0', reload = True)