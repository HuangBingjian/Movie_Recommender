#-*- coding:utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from recommender import rating_movie,recommend_same_type_movie,recommend_user_favorite_movie,recommend_other_favorite_movie

class show_window(QWidget):
    def __init__(self):
        super(show_window, self).__init__()
        self.initUI()

    def initUI(self):
        self.setFixedSize(1050,800)
        self.setWindowTitle('电影打分推荐')

        title_Label = QLabel('机器学习——电影打分推荐',self)
        title_Label.move(400,50)

        author_Label = QLabel('作者：笔尖 / bj',self)
        author_Label.move(900,100)

        validator_user = QIntValidator(1, 6040, self)
        validator_movie = QIntValidator(1, 3592, self)

        fun1_Label = QLabel('打分情况：',self)
        fun1_Label.move(50,500)
        user1_Label = QLabel('用户：',self)
        user1_Label.move(150,500)
        self.user1_Line = QLineEdit('1',self)
        self.user1_Line.move(220,500)
        self.user1_Line.setFixedWidth(80)
        self.user1_Line.setValidator(validator_user)
        mov1_Label = QLabel('电影:',self)
        mov1_Label.move(150,550)
        self.mov1_Line = QLineEdit('1',self)
        self.mov1_Line.move(220,550)
        self.mov1_Line.setFixedWidth(80)
        self.mov1_Line.setValidator(validator_movie)
        fun1_Btn = QPushButton('确定',self)
        fun1_Btn.move(350,550)
        fun1_Btn.clicked.connect(self.rate_movie)
        rate_Label = QLabel('打分：',self)
        rate_Label.move(150,600)
        self.rate_Label = QLabel('——',self)
        self.rate_Label.move(220,600)
        self.rate_Label.setFixedWidth(100)

        fun2_Label = QLabel('相似推荐：',self)
        fun2_Label.move(50,200)
        mov2_Label = QLabel('电影：',self)
        mov2_Label.move(150,200)
        self.mov2_Line = QLineEdit('1',self)
        self.mov2_Line.move(220,200)
        self.mov2_Line.setFixedWidth(80)
        self.mov2_Line.setValidator(validator_movie)
        fun2_Btn = QPushButton('确定',self)
        fun2_Btn.move(350,200)
        fun2_Btn.clicked.connect(self.recommend_same_type)

        fun3_Label = QLabel('猜你喜欢：',self)
        fun3_Label.move(50,300)
        user3_Label = QLabel('用户：',self)
        user3_Label.move(150,300)
        self.user3_Line = QLineEdit('1',self)
        self.user3_Line.move(220,300)
        self.user3_Line.setFixedWidth(80)
        self.user3_Line.setValidator(validator_user)
        fun3_Btn = QPushButton('确定',self)
        fun3_Btn.move(350,300)
        fun3_Btn.clicked.connect(self.recommend_user_favourite)

        fun4_Label = QLabel('志同道合：',self)
        fun4_Label.move(50,400)
        mov4_Label = QLabel('电影:',self)
        mov4_Label.move(150,400)
        self.mov4_Line = QLineEdit('1',self)
        self.mov4_Line.move(220,400)
        self.mov4_Line.setFixedWidth(80)
        self.mov4_Line.setValidator(validator_movie)
        fun4_Btn = QPushButton('确定',self)
        fun4_Btn.move(350,400)
        fun4_Btn.clicked.connect(self.recommend_other_favourite)

        fun5_Label = QLabel('功能设置：',self)
        fun5_Label.move(50,650)
        max_mov_Label = QLabel('推荐电影上限：',self)
        max_mov_Label.move(150,650)
        self.max_mov_Line = QLineEdit('5',self)
        self.max_mov_Line.move(300,650)
        self.max_mov_Line.setFixedWidth(145)
        max_user_Label = QLabel('推荐好友上限：',self)
        max_user_Label.move(150,700)
        self.max_user_Line = QLineEdit('5',self)
        self.max_user_Line.move(300,700)
        self.max_user_Line.setFixedWidth(145)

        result_Label = QLabel('推荐结果：',self)
        result_Label.move(500,150)
        self.result_Edit = QTextEdit(self)
        self.result_Edit.setFixedSize(500,520)
        self.result_Edit.move(500,200)

        self.show()

    def rate_movie(self):
        user = int(self.user1_Line.text())
        mov = int(self.mov1_Line.text())
        result = rating_movie(user, mov)
        self.rate_Label.setText(str(result))

    def recommend_same_type(self):
        mov = int(self.mov2_Line.text())
        result = recommend_same_type_movie(mov,int(self.max_mov_Line.text()))
        for i in result:
            self.result_Edit.append(i)
            self.result_Edit.append('')

    def recommend_user_favourite(self):
        user = int(self.user3_Line.text())
        result = recommend_user_favorite_movie(user,int(self.max_mov_Line.text()))
        for i in result:
            self.result_Edit.append(i)
            self.result_Edit.append('')

    def recommend_other_favourite(self):
        mov = int(self.mov4_Line.text())
        result = recommend_other_favorite_movie(mov,int(self.max_user_Line.text()),int(self.max_mov_Line.text()))
        for i in result:
            self.result_Edit.append(i)
            self.result_Edit.append('')