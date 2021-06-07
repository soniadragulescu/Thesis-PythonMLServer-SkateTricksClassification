if __name__ == '__main__':
    with open('/datas/trainlist_skate.txt', 'w') as f:
        for i in range(70):
            if i < 10:
                f.write('Fails/Fail_00' + str(i) + '.mp4\n')
            else:
                f.write('Fails/Fail_0' + str(i) + '.mp4\n')

        for i in range(81):
            if i < 10:
                f.write('Ollie/Ollie_00' + str(i) + '.mov\n')
            else:
                f.write('Ollie/Ollie_0' + str(i) + '.mov\n')

        for i in range(78):
            if i < 10:
                f.write('Slide/Slide_00' + str(i) + '.mp4\n')
            else:
                f.write('Slide/Slide_0' + str(i) + '.mp4\n')

    with open('/datas/testlist_skate.txt', 'w') as f:
        for i in range(70, 86):
            f.write('Fails/Fail_0' + str(i) + '.mp4\n')

        for i in range(81, 108):
            if i > 99:
                f.write('Ollie/Ollie_' + str(i) + '.mov\n')
            else:
                f.write('Ollie/Ollie_0' + str(i) + '.mov\n')

        for i in range(78, 95):
            f.write('Slide/Slide_0' + str(i) + '.mp4\n')
