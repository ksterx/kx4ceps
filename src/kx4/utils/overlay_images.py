import cv2

fg_dir = r'C:\Users\Lab\Desktop\overlay_imgs'  # foreground images
bg_dir = r'C:\Users\Lab\Desktop\arrws'  # background images
fr = 2.0
size = (1920, 1080)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('C:/Users/Lab/Desktop/estimation_result.mp4', fmt, fr, size)

img_dir = [fg_dir, bg_dir]
for i in range(100):
    fg = fg_dir + '/overlay_img_{:05d}.png'.format(i)
    bg = bg_dir + '/img_{:05d}.jpg'.format(i+1)
    fg = cv2.imread(fg, 1)
    bg = cv2.imread(bg)
    img = cv2.addWeighted(fg, 0.5, bg, 1., 10.)
    cv2.imwrite('C:/Users/Lab/Desktop/estimation/img_{:05d}.png'.format(i+1), img)
    writer.write(img)

writer.release()

print("---end---")
