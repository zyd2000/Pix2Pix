from PIL import Image
import os


path = "D:\\桌面\\实验报告\\深度学习\\picture\\new"  # 最后要加双斜杠，不然会报错
filelist = os.listdir(path)


def main():
    for file in filelist:
        whole_path = os.path.join(path, file)
        img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
        img = img.convert("RGB")  # 将一个4通道转化为rgb三通道
        crop = img.crop((0, 0, 256, 256))
        save_path = 'D:\\桌面\\实验报告\\深度学习\\picture\\nn\\'
        crop.save(save_path + file)


if __name__ == "__main__":
    main()
