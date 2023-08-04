#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
from tkinter import *
from tkinter.font import Font
from tkinter.ttk import *
from tkinter.constants import END

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForMaskedLM

import operator

import warnings

warnings.filterwarnings('ignore')


class Application_ui(Frame):
    # 这个类仅实现界面生成功能，具体事件处理代码在子类Application中。
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('Text Error Correction')
        self.master.geometry('628x438')
        self.createWidgets()

    def createWidgets(self):
        self.top = self.winfo_toplevel()

        self.style = Style()

        self.style.configure('Label1.TLabel', anchor='e', font=('微软雅黑', 14))
        self.Label1 = Label(self.top, text='请输入文本：', style='Label1.TLabel')
        self.Label1.place(relx=0., rely=0.018, relwidth=0.193, relheight=0.075)

        self.Text1Font = Font(font=('微软雅黑', 14))
        self.Text1 = Text(self.top, font=self.Text1Font)
        self.Text1.place(relx=0.191, rely=0.018, relwidth=0.791, relheight=0.313)

        self.style.configure('Label2.TLabel', anchor='e', font=('微软雅黑', 14))
        self.Label2 = Label(self.top, text='纠错后：', style='Label2.TLabel')
        self.Label2.place(relx=0.013, rely=0.365, relwidth=0.167, relheight=0.094)

        self.Text2Font = Font(font=('微软雅黑', 14))
        self.Text2 = Text(self.top, font=self.Text2Font)
        self.Text2.place(relx=0.191, rely=0.365, relwidth=0.791, relheight=0.331)

        self.style.configure('Command1.TButton', font=('微软雅黑', 14))
        self.Command1 = Button(self.top, text='开始', command=self.Command1_Cmd, style='Command1.TButton')
        self.Command1.place(relx=0.369, rely=0.767, relwidth=0.256, relheight=0.185)


class CscModel(nn.Module):
    def __init__(self, tokenizer, device):
        super(CscModel, self).__init__()
        self.bert = base_model
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels[text_labels == 0] = -100  # -100计算损失时会忽略
            text_labels = text_labels.to(self.device)
        else:
            text_labels = None

        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        # 检错输出，纠错输出
        outputs = (
            prob, bert_outputs.logits
        )
        return outputs


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


class Application(Application_ui):
    # 这个类实现具体的事件处理回调函数。界面生成代码在Application_ui中。
    def __init__(self, master=None):
        Application_ui.__init__(self, master)

    def delete_text(self):
        self.Text2.delete('0.0', END)

    def text_get(self):
        return self.Text1.get('0.0', END)

    def Command1_Cmd(self, event=None):
        print(self.text_get())
        if self.text_get() == '':
            print('输入框不能为空')
        else:
            test = self.text_get()
            with torch.no_grad():
                outputs = csc_model(texts=[test])

            _text = tokenizer.decode(torch.argmax(outputs[-1].cpu().detach()[0], dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = _text[:len(test)]
            corrected_text, details = get_errors(corrected_text, test)
            print(test, ' => ', corrected_text, details)

            self.delete_text()
            self.Text2.insert('1.0', corrected_text)


if __name__ == "__main__":
    model_path = 'shibing624/macbert4csc-base-chinese'
    base_model = AutoModelForMaskedLM.from_pretrained('hfl/chinese-macbert-base')

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    learning_rate = 5e-5  # 学习率
    epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csc_model = CscModel(tokenizer=tokenizer, device=device)
    csc_model.to(device)

    csc_model.load_state_dict(torch.load('model.pt'))

    top = Tk()
    Application(top).mainloop()
    try:
        top.destroy()
    except:
        pass
