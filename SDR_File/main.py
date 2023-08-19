import time
from machine import Pin
from machine import UART
import sys
import Lcd1_14driver

LCD = Lcd1_14driver.Lcd1_14()#driver of lcd display
led = Pin(25, machine.Pin.OUT)
led2 = Pin(2, machine.Pin.OUT)
led2.value(0)
led.value(0)
lora = UART(0,baudrate = 9600,tx = Pin(0),rx = Pin(1))


def lcd_border():
        LCD.hline(10,10,220,LCD.blue)
        LCD.hline(10,125,220,LCD.blue)
        LCD.vline(10,10,115,LCD.blue)
        LCD.vline(230,10,115,LCD.blue)
        LCD.lcd_show()

def infoDevice():
        LCD.fill(LCD.white)
        LCD.lcd_show()
        lcd_border()

        LCD.text("SB-COMPONENTS",70,40,LCD.red)
        LCD.text("PICO LORA ",70,60,LCD.red)
        LCD.text("EXPANSION",70,80,LCD.red)
        LCD.lcd_show()
        time.sleep(2)
        LCD.fill(0xFFFF)

        LCD.text("WAITING.....",70,40,LCD.red)
        LCD.lcd_show()
        x = 0
        for y in range(0,1):
             x += 4
             LCD.text("......",125+x,40,LCD.red)
             LCD.lcd_show()
             time.sleep(1)
             

def led_on():
    led.value(1)

def led_off():
    led.value(0)

def receive_data():
    dataRead = lora.readline()
    
    if dataRead is not None:
        led_on()
        dataRead = dataRead.decode("utf-8")
        LCD.text(dataRead,135,80,LCD.red)
        #LCD.text("DATA =",20,60,LCD.blue)
        LCD.text("Recived DATA =",10,80,LCD.blue)
        LCD.lcd_show()
        LCD.fill(0xFFFF)
        print(dataRead)
    return dataRead

infoDevice()

LCD.lcd_show()

while True:
    # read a command from the host
    v = receive_data()
    time.sleep(3)
    #v = sys.stdin.readline().strip()
    
    
    




