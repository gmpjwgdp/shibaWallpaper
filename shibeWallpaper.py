import ctypes
import json
import requests
import os
import io
import random
import cv2
import PySimpleGUI as sg
from PIL import Image, ImageTk, ImageOps
import string
import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class Shibe:
  """犬画像オブジェクト
    :param img: 犬画像の多次元配列データ
    """
  img: np.uint8
  cropped_img: np.uint8 = None
  gray: bool = False
  line: bool = False

  #画像を白黒にする
  def get_gray(self) -> None:
    self.gray = not self.gray
    self.line = False
    return

  #画像を白黒にする
  def get_line(self) -> None:
    self.line = not self.line
    self.gray = False
    return

  #画像を返す
  def get_img(self) -> np.uint8:
    if self.gray:
      return cv2.cvtColor(self.cropped_img, cv2.COLOR_BGR2GRAY)
    elif self.line:
      kernel = np.ones((4,4),np.uint8)
      img_gray = cv2.cvtColor(self.cropped_img, cv2.COLOR_BGR2GRAY)
      dilation = cv2.dilate(img_gray,kernel,iterations = 1)
      diff = cv2.subtract(dilation, img_gray)
      negaposi = 255 - diff
      return negaposi
    else:
      return self.cropped_img



def get_img_data(image:Union[str, np.uint8], first: bool=False) -> ImageTk.PhotoImage:
  """Generate image data using PIL
  """
  if isinstance(image, str):
    img = Image.open(image)
  elif isinstance(image, np.ndarray):
    img = Image.fromarray(image)
  img.thumbnail(size=(640, 400))
  if first:  # tkinter is inactive the first time
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()
  return ImageTk.PhotoImage(img)

def randomname(n: int) -> str:
  '''ランダムなn文字の文字列を返す
  :param n: 文字数
  :return 文字列
  '''
  return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def request(type: str) -> np.uint8:
  '''APIからエンコードされた画像データを取得
  :return data URLから取得したエンコードされたデータ
  '''
  if type == "shibe":
    #shibeAPIにgetリクエスト
    res_api = requests.get('http://shibe.online/api/shibes?count=1&httpsUrls=true')
    #柴犬の画像を返すURLをリスト形式で返される
    url = json.loads(res_api.text)[0]
  if type == "dog":
    res_api = requests.get('https://dog.ceo/api/breeds/image/random')
    #犬の画像を返すURLをmessageキーで返される
    url = json.loads(res_api.text)['message']
  #犬の画像を取得
  res_img = requests.get(url)
  #HTTPレスポンスのバイナリデータ→I/Oオブジェクト→Imageオブジェクトに変換
  image = Image.open(io.BytesIO(res_img.content))
  #多次元配列データに変換
  data = np.array(image)
  return data

def trimming_img(img: np.uint8,resolution :str,width: int=None,height: int=None) -> np.uint8:
  '''解像度に合わせたデータを返却
  :param img 元画像データ
  :param resolution 解像度(xxxx*xxxx)
  :return img_data 解像度を変更した画像データ
  '''
  if resolution == 'デフォルト':
    img_data = img
  else:
    #現在の解像度
    x = img.shape[1]
    y = img.shape[0]
    #セレクトボックスで選択された解像度
    resolution = [int(i) for i in resolution.split("*")]
    target_x = resolution[0]
    target_y = resolution[1]
    sum_target = target_x + target_y
    #目標解像度の比率に対し、長い方の辺の長さを短いほうに合わせる
    if sum_target * x / target_x > sum_target * y / target_y:
      proper_x = int(target_x/target_y * y)
      extra = x - proper_x
      if width is not None:
        extra = int(extra * width/100)
      img_data = img[0:y , extra:proper_x + extra]
    if sum_target * y / target_y > sum_target * x / target_x:
      proper_y = int(target_y/target_x * x)
      extra = y - proper_y
      if height is not None:
        extra = int(extra * height/100)
      img_data = img[extra:proper_y + extra , 0:x]
    else:
      img_data = img
  return img_data

def update_img(window: sg.Window,values: dict,shibe: Shibe) -> None:
  #UI上のスライダーと解像度で画像を編集
  shibe.cropped_img = trimming_img(shibe.img,values['resolution'],values['width'],values['height'])
  #変更後画像を画面上に表示
  window['image'].update(data=get_img_data(shibe.get_img()))
  return


def main():
  sg.theme('Dark Brown')
  shibe = None
  current_directory = os.getcwd()  # カレントディレクトリのパスを取得
  image_path = os.path.join(current_directory, './tempdog.png')  # カレントディレクトリと画像ファイル名を結合

  layout = [[sg.Button('柴犬画像取得', size=(30,4), key="get_shiba"),
              sg.Frame('Edit', [
                [sg.Combo(
                    ['デフォルト','1920*1200','1600*1200','1600*900']
                  , default_value='デフォルト', size=(15,1), enable_events=True, readonly=True, key="resolution")],
                [sg.Button('白黒画像', key="gray")],
                [sg.Button('線画', key="line")]])],
                [sg.Button('犬画像取得', size=(30,1), key="get_dog")],
                [sg.Slider(range=(0.0,100.0), resolution=1.0, enable_events=True, orientation='h', size=(40, None), default_value=50, key="width")],
             [sg.Image(data="", size=(640,400), key="image"),
              sg.Slider(range=(100.0,0.0), resolution=1.0, enable_events=True, orientation='v', size=(20,None), default_value=50, key="height")],
            [sg.Button('この柴犬をデスクトップ画像にする', key="accept")],
            [sg.Button('フォルダに保存', key="save")],
            [sg.Button('柴犬度を計測', key="compare"),sg.Text(text = '',key='text')],
            [sg.Button('壁紙を元に戻す', key="reset")]]

  window = sg.Window('Shibe Wallpaper', layout)

  while True:  # Event Loop
    event,values = window.read()
    if event == 'get_shiba':
      #新規画像を取得
      shibe = Shibe(img = request("shibe"))
      window["text"].update("")

    if event == 'get_dog':
      #新規画像を取得
      shibe = Shibe(request("dog"))
      window["text"].update("")

    if event == 'line':
      if shibe is not None:
        shibe.get_line()
        window['image'].update(data=get_img_data(shibe.get_img()))

    if event == 'gray':
      if shibe is not None:
        #temp画像のモノクロ画像データを取得
        shibe.get_gray()
        #変更後画像を画面上に表示
        window['image'].update(data=get_img_data(shibe.get_img()))

    #UI上で画像に変更を加えた場合、表示する画像を更新する
    if event in ('get_shiba','get_dog','resolution','height','width'):
      if shibe is not None:
        update_img(window,values,shibe)

    if event == 'accept':
      if shibe is not None:
        #壁紙を変更
        Image.fromarray(shibe.get_img()).save(image_path)
        ctypes.windll.user32.SystemParametersInfoW(20, 0, image_path , 0)

    if event == 'save':
      try:
        folder = sg.popup_get_folder("Select a folder")
        #CV2では日本語を含むファイルパスを指定できない
        img = cv2.cvtColor(shibe.get_img(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(folder,randomname(10) + ".png") , img)
        sg.popup('Completed')
      except:
        sg.popup('Error')

    if event == 'compare':
      if shibe is not None:
        #2回目は無視される。非常に重く、ここでしか使わないためここに記述
        from keras.models import load_model
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        model = load_model("./keras_Model.h5", compile=False)
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        image = Image.fromarray(cv2.cvtColor(shibe.get_img(), cv2.COLOR_BGR2RGB))
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        # turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        confidence_score = prediction[0][index]
        if index == 0:
          per = float(confidence_score)  * 100
        else:
          per = (1 - float(confidence_score))  * 100
        text = f"柴犬度は...{per}%です"
        window["text"].update(text)

    if event == 'reset':
      #壁紙をデフォルトに戻す
      ctypes.windll.user32.SystemParametersInfoW(20, 0, None , 0)

    if event == sg.WIN_CLOSED:
      #壁紙変更時、osに処理を投げるため終了を検知が難しく、sleepか最後に消して対応(エラーや強制終了等で残ってしまう)
      if os.path.exists(image_path):
        os.remove(image_path)
      window.close()
      break


if __name__ == "__main__":
  main()
