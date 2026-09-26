import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
from  scipy.ndimage import center_of_mass, shift
model = load_model("digit_recogniser_model.h5")
class DigitRecogniser():
    def __init__ (self):
        self.root = tk.Tk()
        self.root.title("Digit Recogniser")
        self.root.geometry("500x400")

        self.label = tk.Label(self.root, text = "draw a digit", width = 40, height = 1)

        self.clear = tk.Button(self.root, text = "clear", width = 10, command = self.clear_canvas)

        self.predict = tk.Button(self.root, text = "predict", width = 10, command = self.predict_digit)

        self.canvas = tk.Canvas(self.root, width = 300, height = 300, bg = "white")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.label.grid(row = 1, column = 1, columnspan = 2)

        self.canvas.grid(row = 2, column = 1, rowspan = 2)

        self.clear.grid(row = 3, column = 2)

        self.predict.grid(row = 2, column = 2)
        self.image = Image.new("L", (300,300), color = "white")
        self.draw_obj = ImageDraw.draw(self.image)
        self.root.mainloop()
        
    def draw(self, event):
        x,y = event.x, event.y
        r = 8
        self.canvas.create_oval(x -r, y - r, x + r, y + r, fill = "red", outline = "black")
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text = "Draw a digit")
    
    def preproccessimage(self):
        image = self.image.resize((28,28)).convert("L")
        image = np.array(image)
        image = 255 - image
        image = image/255
        cy, cx = center_of_mass(image)
        shiftX = int(np.round(14 - cx))
        shiftY = int(np.round(14 - cy))
        image = shift(image, shift = (shiftY, shiftX), mode = "constant", cval = 0.0)
        image = np.expand_dims(image, axis = 0)
        return image

    def predict_digit(self):
        proccessed_image = self.preproccessimage()
        prediction = self.model.predict(proccessed_image)
        digit = np.argmax(prediction)
        self.label.config(text = f"predction: {digit}" )

app = DigitRecogniser()