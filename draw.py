#coding:utf-8
import numpy
import sys

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

def draw_text_at_center(img, text):
  draw = PIL.ImageDraw.Draw(img)
  draw.font = PIL.ImageFont.truetype(
    "/usr/share/fonts/truetype/takao-mincho/TakaoMincho.ttf", 16)

  img_size = numpy.array(img.size)
  txt_size = numpy.array(draw.font.getsize(text))
  pos = (img_size - txt_size) / 2

  draw.text(pos, text, (0, 100, 255))

img = PIL.Image.new("RGBA", (8*len(sys.argv[1]), 16))
print(sys.argv[1])
text = unicode(sys.argv[1].decode("utf-8"))
draw_text_at_center(img, text)
img.show()
filename = "output_text.ppm"
img.save(filename)
