from PIL import Image

img = Image.open("./test.jpg")
img = img.resize((224, 224))
img.save("./test_224.jpg")