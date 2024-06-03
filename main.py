import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from api_config import get_api
from config import cnf
from logger import get_logger, configure_logger


app = get_api()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_logger()
logger = get_logger("MAIN")

def run():
    logger.info("Starting API")
    logger.info(f"API using device: {cnf.INFERENCE_DEVICE}")
    logger.info(f"API running workers: {cnf.N_WORKERS}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        access_log=True,
        workers=cnf.N_WORKERS,
    )

if __name__ == "__main__":
    run()




# loaded_model.load_state_dict(torch.load('models/model_cifar.pt',  map_location=torch.device("cpu")))
# loaded_model.eval()
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# def predict(image):
#     image.numpy()

#     # move model inputs to cuda, if GPU available
#     image = image.cpu()

#     # get sample outputs
#     output = loaded_model(image)

#     soft_output = torch.softmax(output, 1)
#     score, pred = torch.max(soft_output, 1)

#     return {"pred": classes[int(pred[0])], "score": float(score[0])}


# @app.post("/process_image")
# async def process_image_endpoint(file: UploadFile = File(...)):
#     image = Image.open(BytesIO(await file.read()))
#     transform = torchvision.transforms.ToTensor()
#     tensor_image = transform(image)
#     result = predict(tensor_image)

#     return {"status": "success", "data": result}
