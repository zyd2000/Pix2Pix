from PIL import Image
import os


path = "D:\\桌面\\实验报告\\深度学习\\picture\\new"
filelist = os.listdir(path)

# 将一张图片分成两张图片保存

def main():
    for file in filelist:
        whole_path = os.path.join(path, file)
        img = Image.open(whole_path)
        img = img.convert("RGB")
        crop = img.crop((0, 0, 256, 256))
        save_path = 'D:\\桌面\\实验报告\\深度学习\\picture\\nn\\'
        crop.save(save_path + file)


if __name__ == "__main__":
    main()
