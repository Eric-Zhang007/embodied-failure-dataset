import sys, os

os.environ.setdefault("DISPLAY", ":99")

from ai2thor.controller import Controller

c = Controller(scene="FloorPlan1", agentMode="default", width=300, height=300)

event = c.step(dict(action="RotateRight"))
print("RotateRight success: " + str(event.metadata.get("lastActionSuccess")))

event = c.step(dict(action="Pass"))
print("Pass success: " + str(event.metadata.get("lastActionSuccess")))

screen = c.last_event.frame
print("Frame shape: " + str(screen.shape))

from PIL import Image
img = Image.fromarray(screen)
save_path = "/mnt/c/Users/zjc/embodied-failure-dataset/test_screenshot.png"
img.save(save_path)
print("Screenshot saved to " + save_path)

c.stop()
print("SMOKE_TEST_PASSED")
