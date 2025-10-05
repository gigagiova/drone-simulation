from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, CardMaker, NodePath, TextNode, Texture, TextureStage
from panda3d.core import LineSegs
from direct.gui.OnscreenText import OnscreenText
from drone import Drone
from yolo_detector import YOLOEDetector
import numpy as np
import threading
import random
import os
import cv2


# Configure the window
config = """
win-size 1080 720
window-title Drone Simulation
load-file-type p3assimp
model-cache-dir
"""
loadPrcFileData("", config)


class DroneSimulation(ShowBase):
    def __init__(self):
        super().__init__()

        # Disable default mouse-based camera control
        self.disableMouse()

        # Setup the scene
        self.setup_environment()
        self.setup_scenery()

        # Drone setup
        self.drone = Drone(self.loader, self.render)

        self.setup_cameras()

        # Rich comment: Initialize the YOLOE detector (prompted for "drone") and
        # prepare overlay nodes and state for detection results and concurrency.
        # Prefer Apple's Metal backend on Apple Silicon for faster inference if available.
        # Rich comment: Force CPU for embedding and inference to avoid MPS edge cases until detection works
        preferred_device = "cpu"
        self.detector = YOLOEDetector(
            "yoloe-11s-seg.pt",
            device=preferred_device,
            conf=0.001,
            iou=0.3,
            imgsz=1280,
            visual_ref_path="assets/images/drone.png",
        )
        self.overlay_root = self.pixel2d.attachNewNode("overlays")
        self.overlay_left = self.overlay_root.attachNewNode("left")
        self.overlay_right = self.overlay_root.attachNewNode("right")
        self.latest_results_cam1 = []
        self.latest_results_cam2 = []
        self.detection_in_progress = False
        self.detection_lock = threading.Lock()

        self.taskMgr.add(self.camera_move_task, "camera_move_task")
        self.taskMgr.add(self.drone_move_task, "drone_move_task")
        self.taskMgr.add(self.update_camera2_pos_task, "update_camera2_pos_task")
        self.taskMgr.add(self.update_buffer_cams_task, "update_buffer_cams_task")

        self.coords_text = OnscreenText(
            text="Pos: (0, 0, 0)", pos=(-1.7, 0.9), scale=0.05, align=TextNode.ALeft, mayChange=True
        )
        self.taskMgr.add(self.update_coords_task, "update_coords_task")
        # Rich comment: Run detection each frame to keep overlays fully in sync
        self.taskMgr.add(self.detect_each_frame_task, "detect_each_frame_task")
        # Rich comment: Update overlays every frame (cheap)
        self.taskMgr.add(self.update_overlays_task, "update_overlays_task")

        # Rich comment: Prepare per-frame PNG recording for both cameras. Files are saved under
        # ../video/ as cam1_frame_0001.png and cam2_frame_0001.png, matching the Makefile rule.
        # We clear stale PNGs on startup to avoid mixing runs.
        self._video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "video"))
        self._frame_index = 1
        self._ensure_video_dir()

        self.keyMap = {"w": False, "s": False, "a": False, "d": False}
        self.accept("w", self.setKey, ["w", True])
        self.accept("s", self.setKey, ["s", True])
        self.accept("a", self.setKey, ["a", True])
        self.accept("d", self.setKey, ["d", True])
        self.accept("w-up", self.setKey, ["w", False])
        self.accept("s-up", self.setKey, ["s", False])
        self.accept("a-up", self.setKey, ["a", False])
        self.accept("d-up", self.setKey, ["d", False])

        # Rich comment: Removed periodic background detection; using per-frame detection instead

        """# Ensure the video directory exists
        if not os.path.exists("video"):
            os.makedirs("video")

        # Start recording for the first 10 seconds
        self.taskMgr.doMethodLater(0, self.start_recording, "start_recording")
        self.taskMgr.doMethodLater(10, self.stop_recording, "stop_recording")"""

    def start_recording(self, task):
        """Starts recording the simulation."""
        # Record from the first buffer
        self.movie(
            namePrefix="video/cam1_frame",
            duration=10.0,
            fps=30,
            format="png",
            sd=4,
            source=self.buffer1,
        )
        # Record from the second buffer
        self.movie(
            namePrefix="video/cam2_frame",
            duration=10.0,
            fps=30,
            format="png",
            sd=4,
            source=self.buffer2,
        )
        print("Started recording from both cameras...")
        return task.done

    def stop_recording(self, task):
        """Stops recording the simulation."""
        print("...Finished recording from both cameras.")
        return task.done

    def drone_move_task(self, task):
        """Moves the drone based on its internal logic."""
        self.drone.update(self.camera, task.dt)
        return task.cont

    def camera_move_task(self, task):
        dt = task.dt
        speed = 20000
        rot_speed = 100000

        # Get the forward vector of the camera in world space.
        forward_vec = self.camera.getQuat(self.render).getForward()
        # Project the vector onto the XY plane.
        forward_vec.setZ(0)
        # Normalize the vector to ensure consistent speed.
        forward_vec.normalize()

        # Move forward/backward
        if self.keyMap["w"]:
            self.camera.setPos(self.camera.getPos() + forward_vec * speed * dt)
        if self.keyMap["s"]:
            self.camera.setPos(self.camera.getPos() - forward_vec * speed * dt)

        # Rotate left/right
        if self.keyMap["a"]:
            self.camera.setH(self.camera.getH() + rot_speed * dt)
        if self.keyMap["d"]:
            self.camera.setH(self.camera.getH() - rot_speed * dt)

        return task.cont

    def update_camera2_pos_task(self, task):
        """Keeps the second camera to the right of the first camera."""
        # The quat represents the rotation of the camera.
        # getRight() gives a vector pointing to the right of the camera.
        right_vec = self.camera.getQuat(self.render).getRight()
        # The new position is 1 meter to the right of the first camera.
        self.camera2.setPos(self.camera.getPos() + right_vec * 1)
        # Keep the orientation of the second camera the same as the first.
        self.camera2.setHpr(self.camera.getHpr())
        return task.cont

    def update_buffer_cams_task(self, task):
        """Updates the buffer cameras to match the main cameras."""
        self.cam1_buffer.setPosHpr(
            self.camera.getPos(),
            self.camera.getHpr(),
        )
        self.cam2_buffer.setPosHpr(
            self.camera2.getPos(),
            self.camera2.getHpr(),
        )
        return task.cont

    def update_coords_task(self, task):
        pos = self.camera.getPos()
        self.coords_text.setText(f"Pos: ({pos.getX():.2f}, {pos.getY():.2f}, {pos.getZ():.2f})")
        return task.cont

    def setKey(self, key, value):
        self.keyMap[key] = value

    def setup_scenery(self):
        """
        Populates the world with trees.

        This method instances a tree model at random locations on the ground plane,
        respecting a given density.
        """
        # Define the area for placing trees, matching the ground plane size of the skybox.
        # The skybox ground plane is 400x400 units.
        area_side = 400
        # The desired density of trees is 4 per 100 square meters.
        density_per_100_sq_m = 1

        # Calculate the total number of trees to place based on the area and density.
        total_area = area_side * area_side
        num_trees = int((total_area / 100) * density_per_100_sq_m)

        # Load the tree model from the assets folder.
        tree_model = self.loader.loadModel("../assets/models/Tree.obj")
        tree_model.setTransparency(True)

        # Get the dimensions of the model to calculate the correct scale.
        min_bounds, max_bounds = tree_model.getTightBounds()
        # The height of the model is the difference in the Y-axis bounds, assuming the model is oriented along Y.
        model_height = max_bounds.y - min_bounds.y

        # Create a parent node for all the trees. This helps in keeping the scene graph organized.
        scenery_node = self.render.attachNewNode("scenery")

        # Rich comment: Define a circular exclusion zone (10 meters radius) centered at the world origin
        # on the XY plane. Trees will not be placed inside this zone to keep a clear area around (0, 0).
        exclusion_center_x = 0.0
        exclusion_center_y = 0.0
        exclusion_radius_m = 20.0
        exclusion_radius_sq = exclusion_radius_m * exclusion_radius_m

        # Loop to create and place each tree instance.
        for _ in range(num_trees):
            # An empty node is created as a placeholder for the tree instance.
            tree_instance = scenery_node.attachNewNode("tree_instance")
            # The tree model is instanced to the placeholder. This is efficient for rendering many copies.
            tree_model.instanceTo(tree_instance)

            # The desired height for this tree instance, normally distributed.
            desired_height = random.gauss(5, 0.5)
            # Calculate the scale factor required to achieve the desired height.
            scale_factor = desired_height / model_height

            # Rich comment: Rejection sample a random XY position until it is outside the exclusion radius.
            # This preserves the total tree count while ensuring a clear 10m circle around (0, 0).
            while True:
                x = random.uniform(-area_side / 2, area_side / 2)
                y = random.uniform(-area_side / 2, area_side / 2)
                dx = x - exclusion_center_x
                dy = y - exclusion_center_y
                if (dx * dx + dy * dy) >= exclusion_radius_sq:
                    break
            # The tree is placed on the ground, accounting for the model's origin offset.
            z = -min_bounds.y * scale_factor
            tree_instance.setPos(x, y, z)

            # Set the calculated scale to achieve the desired height.
            tree_instance.setScale(scale_factor)
            # Rotate the tree to be upright (90 degrees on pitch) and give it a random heading.
            tree_instance.setHpr(random.uniform(0, 360), 90, 0)

    def setup_environment(self):
        """Sets up the skybox environment."""
        self.skybox = self.create_skybox()
        self.skybox.reparentTo(self.render)
        # Scale the skybox to be large enough
        self.skybox.setScale(200)  # Set side length to 400m (since card is 2x2)
        # Position the skybox so the ground is at Z=0
        self.skybox.setPos(0, 0, 200)

    def create_skybox(self):
        """Creates a skybox with a green ground and sky textures."""
        skybox = NodePath("skybox")
        sky_texture = self.loader.loadTexture("../assets/textures/sky.png")
        grass_texture = self.loader.loadTexture("../assets/textures/grass.png")
        grass_texture.setWrapU(Texture.WM_repeat)
        grass_texture.setWrapV(Texture.WM_repeat)

        # Create the 6 faces of the cube using CardMaker
        card_maker = CardMaker("skybox_face")
        card_maker.setFrame(-1, 1, -1, 1)

        faces = []
        for i in range(6):
            node = NodePath(card_maker.generate())
            if i < 5:  # Not the bottom face
                node.setTexture(sky_texture)
            faces.append(node)

        # Reposition and reorient the faces to form a cube
        # Front face
        faces[0].reparentTo(skybox)
        faces[0].setPos(0, -1, 0)
        faces[0].setHpr(0, 0, 0)

        # Back face
        faces[1].reparentTo(skybox)
        faces[1].setPos(0, 1, 0)
        faces[1].setHpr(180, 0, 0)

        # Right face
        faces[2].reparentTo(skybox)
        faces[2].setPos(1, 0, 0)
        faces[2].setHpr(-90, 0, 0)

        # Left face
        faces[3].reparentTo(skybox)
        faces[3].setPos(-1, 0, 0)
        faces[3].setHpr(90, 0, 0)

        # Top face
        faces[4].reparentTo(skybox)
        faces[4].setPos(0, 0, 1)
        faces[4].setHpr(0, -90, 0)

        # Bottom face (ground)
        faces[5].reparentTo(skybox)
        faces[5].setPos(0, 0, -1)
        faces[5].setHpr(0, 90, 0)
        faces[5].setTexture(grass_texture)
        faces[5].setTexScale(TextureStage.getDefault(), 100, 100)

        # Set render properties for the skybox to ensure it's drawn correctly
        skybox.setShaderOff()
        skybox.setBin("background", 0)
        skybox.setDepthWrite(False)
        skybox.setDepthTest(False)
        skybox.setLightOff()
        skybox.setTwoSided(True)

        return skybox

    def setup_cameras(self):
        """Sets up the cameras and their display regions."""
        # Disable the default camera and control
        self.disableMouse()
        self.cam.node().setActive(False)

        # Create the first camera for window display
        self.camera = self.makeCamera(self.win, displayRegion=(0, 0.5, 0, 1))
        self.camera.reparentTo(self.render)
        self.camera.setPos(0, 0, 2)
        self.camera.setHpr(0, 5, 0)

        # Create the second camera for window display
        self.camera2 = self.makeCamera(self.win, displayRegion=(0.5, 1, 0, 1))
        self.camera2.reparentTo(self.render)

        # Create offscreen buffers for recording
        self.buffer1 = self.win.makeTextureBuffer("Buffer1", 1080, 720)
        self.buffer2 = self.win.makeTextureBuffer("Buffer2", 1080, 720)

        # Create cameras for the offscreen buffers
        self.cam1_buffer = self.makeCamera(self.buffer1)
        self.cam1_buffer.reparentTo(self.render)
        self.cam2_buffer = self.makeCamera(self.buffer2)
        self.cam2_buffer.reparentTo(self.render)

    # Rich comment: Helper to convert a render buffer into a numpy RGB image suitable for YOLO.
    def buffer_to_numpy_rgb(self, buffer):
        # Rich comment: Pull pixel data for the buffer's color texture into system RAM and convert to numpy.
        # Important: We refresh the RAM image every frame so we don't repeatedly read the first frame only.
        tex = buffer.getTexture()
        if tex is None:
            return None
        try:
            # Rich comment: Force a RAM copy from GPU each frame; prefer the buffer's GSG
            gsg = buffer.getGsg() if hasattr(buffer, "getGsg") else None
            if gsg is None:
                gsg = self.win.getGsg()
            self.graphicsEngine.extractTextureData(tex, gsg)
            if not tex.hasRamImage():
                return None
            # Rich comment: Textures may be allocated at power-of-two sizes (e.g., 2048x1024)
            # while the actual rendered buffer size is smaller (e.g., 1080x720). Crop to buffer size.
            tex_w = tex.getXSize()
            tex_h = tex.getYSize()
            buf_w = buffer.getXSize() if hasattr(buffer, "getXSize") else tex_w
            buf_h = buffer.getYSize() if hasattr(buffer, "getYSize") else tex_h
            num_components = tex.getNumComponents()
            ram = tex.getRamImage()
            if ram is None:
                return None
            arr = np.frombuffer(ram, dtype=np.uint8)
            if arr.size != tex_w * tex_h * num_components:
                return None
            # Rich comment: Reshape, crop to actual buffer rect, then flip vertically
            arr = arr.reshape((tex_h, tex_w, num_components))
            arr = arr[:buf_h, :buf_w, :]
            if num_components >= 3:
                # Rich comment: GPU RAM typically comes as BGR(A) — convert to RGB for model and debug
                bgr = arr[:, :, :3]
                rgb = bgr[:, :, ::-1]
            else:
                rgb = np.repeat(arr, 3, axis=2)
            rgb = np.flipud(rgb.copy())
            return rgb
        except Exception:
            return None

    # Rich comment: Periodic detection task. Captures both camera buffers, runs YOLOE
    # in a background thread, and reschedules itself.
    def run_detection_task(self, task):
        if not self.detection_in_progress:
            left_img = self.buffer_to_numpy_rgb(self.buffer1)
            right_img = self.buffer_to_numpy_rgb(self.buffer2)

            def worker():
                left_results = self.detector.predict(left_img) if left_img is not None else []
                right_results = self.detector.predict(right_img) if right_img is not None else []
                with self.detection_lock:
                    self.latest_results_cam1 = left_results
                    self.latest_results_cam2 = right_results
                self.detection_in_progress = False

            self.detection_in_progress = True
            threading.Thread(target=worker, daemon=True).start()

        # Rich comment: Reschedule to run again after a short delay (balance cost vs responsiveness)
        task.delayTime = 0.2
        return task.again

    # Rich comment: Per-frame detection. Captures both buffers, runs YOLO predict synchronously,
    # and stores results for overlay drawing.
    def detect_each_frame_task(self, task):
        left_img = self.buffer_to_numpy_rgb(self.buffer1)
        right_img = self.buffer_to_numpy_rgb(self.buffer2)

        # Rich comment: Run detector on the RGB images and cache results for on-screen overlays
        left_results = self.detector.predict(left_img) if left_img is not None else []
        right_results = self.detector.predict(right_img) if right_img is not None else []

        with self.detection_lock:
            self.latest_results_cam1 = left_results
            self.latest_results_cam2 = right_results

        # Rich comment: Persist both camera views as PNGs with thick boxes baked in, so
        # `make video` can stitch them via ffmpeg. Filenames are zero-padded to 4 digits.
        if left_img is not None:
            self._save_png_frame("cam1", left_img, left_results)
        if right_img is not None:
            self._save_png_frame("cam2", right_img, right_results)

        # Rich comment: Advance frame index after both saves attempt
        self._frame_index += 1

        return task.cont

    # Rich comment: Clear and redraw simple rectangular overlays for each detection.
    def update_overlays_task(self, task):
        # Clear previous overlays
        self.overlay_left.node().removeAllChildren()
        self.overlay_right.node().removeAllChildren()

        # Window metrics and display region splits
        win_props = self.win.getProperties()
        win_w = win_props.getXSize()
        half_w = int(win_w * 0.5)

        # Draw rectangles for left camera
        with self.detection_lock:
            left_results = list(self.latest_results_cam1)
            right_results = list(self.latest_results_cam2)

        if left_results:
            self._draw_boxes(self.overlay_left, left_results, x_offset_px=0, y_offset_px=0)
        if right_results:
            self._draw_boxes(self.overlay_right, right_results, x_offset_px=half_w, y_offset_px=0)

        return task.cont

    # Rich comment: Build a rectangle from XYXY in pixel coordinates using LineSegs under pixel2d
    def _draw_boxes(self, parent_np, results, x_offset_px, y_offset_px):
        segs = LineSegs()
        segs.setThickness(2)
        # Red for boxes
        segs.setColor(1, 0, 0, 1)

        for det in results:
            x1, y1, x2, y2 = det.xyxy
            # Convert from image coordinates (origin top-left) to pixel2d (origin bottom-left)
            # Query the left buffer's size; both buffers share the same size
            tex = self.buffer1.getTexture()
            height = tex.getYSize() if tex else 720
            by1 = height - y1
            by2 = height - y2

            # Rectangle lines in pixel space
            segs.moveTo(x_offset_px + x1, y_offset_px + by1, 0)
            segs.drawTo(x_offset_px + x2, y_offset_px + by1, 0)
            segs.drawTo(x_offset_px + x2, y_offset_px + by2, 0)
            segs.drawTo(x_offset_px + x1, y_offset_px + by2, 0)
            segs.drawTo(x_offset_px + x1, y_offset_px + by1, 0)

        node = segs.create()
        parent_np.attachNewNode(node)

    # Rich comment: Ensure the video directory exists and is clean of old PNGs.
    def _ensure_video_dir(self):
        try:
            if not os.path.exists(self._video_dir):
                os.makedirs(self._video_dir, exist_ok=True)
            else:
                for name in os.listdir(self._video_dir):
                    if name.endswith(".png") and (name.startswith("cam1_frame_") or name.startswith("cam2_frame_")):
                        try:
                            os.remove(os.path.join(self._video_dir, name))
                        except Exception:
                            pass
        except Exception:
            pass

    # Rich comment: Save one RGB frame to a PNG with fat red rectangles drawn for each detection.
    # The input is HxWx3 uint8 RGB; we convert to BGR for OpenCV before saving.
    def _save_png_frame(self, cam_prefix, image_rgb, detections):
        try:
            h, w, _ = image_rgb.shape
            # Expect 1080x720 buffers, but do not enforce; save whatever size comes from the buffer.
            frame_bgr = image_rgb[:, :, ::-1].copy()
            thickness = 2
            color_bgr = (0, 0, 255)

            for det in detections:
                x1, y1, x2, y2 = det.xyxy
                x1i = max(0, min(int(round(x1)), w - 1))
                y1i = max(0, min(int(round(y1)), h - 1))
                x2i = max(0, min(int(round(x2)), w - 1))
                y2i = max(0, min(int(round(y2)), h - 1))
                if x2i > x1i and y2i > y1i:
                    cv2.rectangle(frame_bgr, (x1i, y1i), (x2i, y2i), color_bgr, thickness)

            filename = f"{cam_prefix}_frame_{self._frame_index:04d}.png"
            out_path = os.path.join(self._video_dir, filename)
            cv2.imwrite(out_path, frame_bgr)
        except Exception:
            # Keep the simulation responsive even if disk IO or encoding fails
            pass


app = DroneSimulation()
app.run()
