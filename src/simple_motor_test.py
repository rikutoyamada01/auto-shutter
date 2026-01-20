import time
from gpiozero import Robot

def test_motors():
    print("Initializing Robot with left=(17,18), right=(19,20)...")
    try:
        # Check pin factory to see what is being used (e.g., pigpio, rpigpio, native)
        import gpiozero
        print(f"Pin Factory: {gpiozero.Device.pin_factory}")
        
        robot = Robot(left=(17,18), right=(19,20))
    except Exception as e:
        print(f"Failed to initialize Robot: {e}")
        return

    print("--- Test Start ---")
    
    print("1. Forward (Left & Right Forward)")
    robot.forward(0.4)
    time.sleep(1)
    robot.stop()
    time.sleep(0.5)

    print("2. Backward (Left & Right Backward)")
    robot.backward(0.4)
    time.sleep(1)
    robot.stop()
    time.sleep(0.5)
    
    print("3. Left Motor Only (Forward)")
    robot.left_motor.forward(0.4)
    time.sleep(1)
    robot.left_motor.stop()
    time.sleep(0.5)

    print("4. Right Motor Only (Forward)")
    robot.right_motor.forward(0.4)
    time.sleep(1)
    robot.right_motor.stop()
    
    print("--- Test Complete ---")

if __name__ == "__main__":
    test_motors()
