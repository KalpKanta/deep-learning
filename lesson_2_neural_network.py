import tkinter as tk

class DigitRecogniser():
    def __init__ (self):
        self.root = tk.Tk()
        self.root.title("Digit Recogniser")
        self.root.geometry("500x400")

        self.label = tk.Label(self.root, text = "draw a digit", width = 40, height = 1)

        self.clear = tk.Button(self.root, text = "clear", width = 10)

        self.predict = tk.Button(self.root, text = "predict", width = 10)

        self.canvas = tk.Canvas(self.root, width = 300, height = 300, bg = "white")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.label.grid(row = 1, column = 1, columnspan = 2)

        self.canvas.grid(row = 2, column = 1, rowspan = 2)

        self.clear.grid(row = 3, column = 2)

        self.predict.grid(row = 2, column = 2)

        self.root.mainloop()

    def draw(self, event):
        x,y = event.x, event.y
        r = 8
        self.canvas.create_oval(x -r, y - r, x + r, y + r, fill = "red", outline = "black")
        

app = DigitRecogniser()