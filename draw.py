import numpy

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

def draw_text_at_center(img, text):
  draw = PIL.ImageDraw.Draw(img)
  draw.font = PIL.ImageFont.truetype(
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)

  img_size = numpy.array(img.size)
  txt_size = numpy.array(draw.font.getsize(text))
  pos = (img_size - txt_size) / 2

  draw.text(pos, text, (255, 255, 255))

img = PIL.Image.new("RGBA", (32, 16))
text = "Hello, world!"
draw_text_at_center(img, text)
img.show()
filename = "output_text.ppm"
img.save(filename)
