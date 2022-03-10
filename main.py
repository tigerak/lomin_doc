from util import Btn_img_out

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import *
from tkinter import filedialog

import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

import warnings
warnings.filterwarnings('ignore')

root = Tk()
root.title('Differentiation is Gradient')
root.geometry('800x600')
root.resizable(False, False) 

# s = ttk.Style()
# s.theme_use('clam')


### 메뉴 ###
menu = Menu(root)

menu_file = Menu(menu, tearoff=0)
menu_file.add_separator()
menu_file.add_command(label='프로그램 종료', command=root.quit)

menu_edit = Menu(menu, tearoff=0)
menu_edit.add_radiobutton(label='편집 없음', state='disable')
menu_edit.add_radiobutton(label='편집 있음', state='disable')

def mail_alt():
    msgbox.showinfo('이봐요', '문제가 있나요?\ntigerakyb@gmail.com\n메일을 보내주세요 :)')
    return

menu_help = Menu(menu, tearoff=0)
menu_help.add_command(label='버그 신고', command=mail_alt)
menu_help.add_command(label='문의 메일', command=mail_alt)
menu_help.add_separator()
menu_help.add_command(label='Version_1.0')

menu.add_cascade(label='File', menu=menu_file)
menu.add_cascade(label='Edit', menu=menu_edit)
menu.add_cascade(label='Help', menu=menu_help)

root.config(menu=menu)
    
    
    
### 이미지 파일 선택 ###
frame_img_lode = LabelFrame(root, text='이미지 파일 선택', bd=1)
frame_img_lode.grid(row=0, column=0, padx=5, pady=5, sticky=N+E+W+S)

entry_text_address = Entry(frame_img_lode)
entry_text_address.insert(0, '이미지 파일을 선택해주세요.')
entry_text_address.grid(row=0, column=0, columnspan=7, 
                        padx=5, pady=5, sticky=W+E)


# 분석할 이미지 파일 추가 
def btn_img_load_cmd():
    entry_text_address.delete(0, END)
    files = filedialog.askopenfilenames(title='이미지 파일을 선택해주세요',
                                        filetypes=(('모든 파일', '*.*'),
                                                   ('png 파일', '*.png'),
                                                   ('jpg 파일', '*.jpg'),
                                                   ('gif 파일', '*.gif'),
                                                   ('bpm 파일', '*.bpm')),
                                        initialdir=r'D:/lomin/data/test_data')
    entry_text_address.insert(END, files)
    file_name = entry_text_address.get()
    
    Btn_img_out().ferst_show(file_name, frame_input_img)
    
btn_txt_load = Button(frame_img_lode, height=1, 
                      text='찾아보기',
                      command=btn_img_load_cmd)
btn_txt_load.grid(row=0, column=7, padx=5, pady=5, sticky=W+E)


# 분석 버튼
def btn_date_save_cmd():
    file_name = entry_text_address.get()
    
    Btn_img_out().forward(file_name, frame_output_img)
    
                
btn_date_save = Button(frame_img_lode, width=25, height=2, 
                      text='분석 시작',
                      command=btn_date_save_cmd)
btn_date_save.grid(row=1, column=0, columnspan=4, 
                   padx=5, pady=5, sticky=W+E)


# 문서 탐색 및 분석 버튼
from yolov5.lomin import main, parse_opt

import matplotlib.pyplot as plt
import cv2
def btn_img_detect_cmd():
    file_name = entry_text_address.get()
    promp = (['--source', file_name, 
              '--weights', './yolov5/runs/train/doc2006/weights/best.pt', 
              '--project', './data/img_data',
            #   '--nosave',
              '--save-crop']
    )
    opt = parse_opt(promp)
    crop_img_list = main(opt)
    
    crop_size_list = {}
    for i, c in enumerate(crop_img_list):
        crop_size = len(c) * len(c[0])
        crop_size_list[i] = crop_size
        
    sel_cro_num = max(crop_size_list, key=crop_size_list.get)
    
    fig = plt.figure(figsize=(12,12))
    img = cv2.cvtColor(crop_img_list[sel_cro_num], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    fig.savefig('demo.png', bbox_inches='tight')
    plt.close()
    
    Btn_img_out().forward('D:/lomin/demo.png', frame_output_img)

    
    
btn_txt_load = Button(frame_img_lode, width=25, height=2, 
                      text='문서 탐색하여 분석',
                      command=btn_img_detect_cmd)
btn_txt_load.grid(row=1, column=4, columnspan=4,
                  padx=5, pady=5, sticky=W+E)




### 입력된 이미지 보기 ###
frame_input_img = LabelFrame(root, text='! 분석할 이미지 !', bd=1)
frame_input_img.grid(row=1, column=0, 
                     padx=5, pady=5, sticky=N+E+W+S)

# 이미지 출력
file_name = None
Btn_img_out().ferst_show(file_name, frame_input_img)




### 결과 출력 ###
frame_output_img = LabelFrame(root, text='! 분석된 이미지 !', 
                         bd=1, width=400)
frame_output_img.grid(row=1, column=1, rowspan=2,
                      padx=5, pady=5, sticky=N+E+W+S)

# 이미지 출력
Btn_img_out().ferst_show(file_name, frame_output_img)














##########################################################


root.mainloop()