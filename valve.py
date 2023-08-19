from pyfirmata import Arduino, util, OUTPUT
def init():
    global board
    board = Arduino('COM12')#Auto search Arduino 2560 board
    board.digital[7].mode = OUTPUT
    return 1


def trig(sta):
    if sta==0:
        board.digital[7].write(0)
        return 0
    elif sta==1:
        board.digital[7].write(1)
        return 1