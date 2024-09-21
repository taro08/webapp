from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

# fastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 予測モデル読み込み
with open("model.pickle", "rb") as f:
    model = pickle.load(f)
with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# 最初のページ
@app.get("/", response_class=HTMLResponse)
async def root(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 予測結果のページ
@app.get("/predict_price/", response_class=HTMLResponse)
async def predict_price(request:Request,
        station_dist:float=0.0, conveni_num:float=0.0,
        milk_consum:float=0.0, income:float=0.0, city_flag:int=0):

    # モデルの入力
    x = [[station_dist, conveni_num, milk_consum, income, city_flag]]
    # 予測
    predicted = model.predict(scaler.transform(x))
    # 予測結果を取り出し、小数点以下6桁で丸める
    predicted_price = round(predicted[0], 6)
    return templates.TemplateResponse("result.html",
                                {"request": request,
                                 "predicted_price": predicted_price})
