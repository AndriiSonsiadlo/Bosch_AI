#Copyright (C) 2020 BardzoFajnyZespol

from kivy.uix.screenmanager import Screen
import csv
import matplotlib.pyplot as plt
from datetime import date
import os.path
import os

class Statistics(Screen):

    def update_left_plot(self, source):
        self.ids.left_plot.source = source
        self.ids.left_plot.reload()
        print('updated plot', source)


    def update_right_plot(self, source):
        self.ids.right_plot.source = source
        self.ids.right_plot.reload()
        print('updated plot', source)

    def data_to_file_generator(self):

        if os.path.exists('basic_data') == True:

            total = 0
            NOK = 0
            too_long = 0
            analysis_time = 0

            with open('basic_data', 'r') as csvfile:
                file_rd = csv.reader(csvfile, delimiter=',')
                current_hour = 0
                for row in file_rd:
                    T1 = (int(row[0]))
                    timing = (float(row[1]))
                    result = (str(row[2]))

                    if T1 == current_hour:

                        total += 1
                        analysis_time += timing

                        if result == "False":
                            NOK += 1
                        if timing > 5:
                            too_long += 1

                        current_hour = T1


                    else:

                        if total != 0:
                            today = date.today()
                            d = today.strftime("%d/%m/%Y")

                            data = [d, current_hour, float(NOK / total * 100), float(too_long / total * 100),
                                    float(analysis_time / total)]

                            with open('data_file', "a", newline='') as gen_file:
                                writing = csv.writer(gen_file, delimiter=',')
                                writing.writerow(data)

                            total = 1
                            NOK = 0
                            too_long = 0
                            analysis_time = 0

                            if result == "False":
                                NOK += 1
                            if timing > 5:
                                too_long += 1

                            analysis_time += timing
                            current_hour = T1

                        else:

                            total += 1
                            analysis_time += timing

                            if result == "False":
                                NOK += 1
                            if timing > 5:
                                too_long += 1

                            current_hour = T1

            today = date.today()
            d = today.strftime("%d/%m/%Y")

            data = [d, current_hour, float(NOK / total * 100), float(too_long / total * 100),
                    float(analysis_time / total)]

            with open('data_file', "a", newline='') as gen_file:
                writing = csv.writer(gen_file, delimiter=',')
                writing.writerow(data)
        else:
            print("basic_data: no such a file")

    def chart(self):

        if os.path.exists('data_file') == True:

            x = []
            y = []
            z = []
            k = []

            with open('data_file', 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                for row in plots:
                    x.append(float(row[1]))
                    y.append(float(row[3]))
                    z.append(float(row[2]))
                    k.append(float(row[4]))


            plt.bar(x, k, label='Average analysis timings')
            plt.legend()
            plt.xlabel('Hour \n[h]')
            plt.ylabel('Average analysis timings \n[s]')
            plt.title('Average analysis timings \n(per hour)')
            plt.tight_layout()
            plt.savefig('Images/average.png')
            plt.clf()

            plt.bar(x, y, label='Too long analysis timings')
            plt.bar(x, z, label='NOK pictures')
            plt.legend()
            plt.xlabel('Hour \n[h]')
            plt.ylabel('Percentage \n[%]')
            plt.title('Percentage of too long analysis timings or NOK pictures\n(per hour)')
            plt.tight_layout()
            plt.savefig('Images/timings&NOK.png')
            plt.clf()

            os.remove('data_file')
            os.remove('basic_data')

        else:
            print("data_file: no such a file")




