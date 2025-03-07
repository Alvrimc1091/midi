from gpiozero import LED
import time


led_red = LED(26) # Rojo -> Alerta
led_green = LED(6) # Verde -> Trigo
led_blue = LED(22) # Azul -> Poroto
led_white = LED(5) # Blanco -> Vacío
led_yellow = LED(27) # Amarillo -> Maíz

time_slp = 0.2
times = 3

# List of LEDs to blink
leds = [led_blue, led_yellow, led_green, led_white, led_red]

def encender_led(led):
    """Enciende un LED durante 2 segundos."""
    if led is not None:
        led.on()
        time.sleep(2)
        led.off()

def secuencia_led_inicializacion(time_var):

    for i in range(3):
        
        time_var = time_slp
        led_blue.on()
        time.sleep(time_slp)
        led_blue.off()
        led_yellow.on()
        time.sleep(time_slp)
        led_yellow.off()
        led_green.on()
        time.sleep(time_slp)
        led_green.off()
        led_red.on()
        time.sleep(time_slp)
        led_red.off()
        led_white.on()
        time.sleep(time_slp)
        led_white.off()


def secuencia_encendido_leds():
        
    led_blue.on()
    time.sleep(time_slp)
    led_blue.off()
    led_yellow.on()
    time.sleep(time_slp)
    led_yellow.off()
    led_green.on()
    time.sleep(time_slp)
    led_green.off()
    led_red.on()
    time.sleep(time_slp)
    led_red.off()
    led_white.on()
    time.sleep(time_slp)
    led_white.off()

def secuencia_encendido_leds_inverso():
        
    led_white.on()
    time.sleep(time_slp)
    led_white.off()
    led_red.on()
    time.sleep(time_slp)
    led_red.off()
    led_green.on()
    time.sleep(time_slp)
    led_green.off()
    led_yellow.on()
    time.sleep(time_slp)
    led_yellow.off()
    led_blue.on()
    time.sleep(time_slp)
    led_blue.off()

def secuencia_encendido_leds_cruzado():
    
    led_blue.on()
    time.sleep(time_slp)
    led_blue.off()
    led_red.on()
    time.sleep(time_slp)
    led_red.off()
    led_yellow.on()
    time.sleep(time_slp)
    led_yellow.off()
    led_white.on()
    time.sleep(time_slp)
    led_white.off()
    led_green.on()
    time.sleep(time_slp)
    led_green.off()

def secuencia_encendido_leds_cruzado_inverso():

    led_green.on()
    time.sleep(time_slp)
    led_green.off()
    led_white.on()
    time.sleep(time_slp)
    led_white.off()
    led_yellow.on()
    time.sleep(time_slp)
    led_yellow.off()
    led_red.on()
    time.sleep(time_slp)
    led_red.off()
    led_blue.on()
    time.sleep(time_slp)
    led_blue.off()

def blink_leds(counts):
    
    counts = times

    for i in range(counts):
        led_red.on()
        led_blue.on()
        led_green.on()
        led_yellow.on()
        led_white.on()
        time.sleep(0.5)
        led_red.off()
        led_blue.off()
        led_green.off()
        led_white.off()
        led_yellow.off()
        time.sleep(0.5)

def blink_leds_sequentially():
    while True:
        for led in leds:
            led.on()
            time.sleep(0.5)  # LED on for 1 second
            led.off()
            time.sleep(0.3)  # Pause for 0.5 seconds before the next LED
