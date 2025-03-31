from pynput.keyboard import Controller

keyboard = Controller()

def next_slide():
    keyboard.press('pagedown')
    keyboard.release('pagedown')

def previous_slide():
    keyboard.press('pageup')
    keyboard.release('pageup')

if __name__ == "__main__":
    print("Press 'n' for Next Slide, 'p' for Previous Slide, 'q' to Quit")
    while True:
        key = keyboard.read_event().name
        if key == 'n':
            next_slide()
            print("Next Slide")
        elif key == 'p':
            previous_slide()
            print("Previous Slide")
        elif key == 'q':
            print("Exiting...")
            break
