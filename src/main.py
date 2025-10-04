from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, CardMaker, NodePath, TextNode, Texture, TextureStage
from direct.gui.OnscreenText import OnscreenText
from drone import Drone
import random


# Configure the window
config = """
win-size 1280 720
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

        # Drone setup, starting at the origin 2m above the ground
        self.drone = self.render.attachNewNode("drone")
        Drone().mesh.reparentTo(self.drone)
        self.drone.setPos(0, 0, 2)

        self.setup_cameras()

        self.taskMgr.add(self.camera_move_task, "camera_move_task")

        self.coords_text = OnscreenText(
            text="Pos: (0, 0, 0)", pos=(-1.7, 0.9), scale=0.05, align=TextNode.ALeft, mayChange=True
        )
        self.taskMgr.add(self.update_coords_task, "update_coords_task")

        self.keyMap = {"w": False, "s": False, "a": False, "d": False}
        self.accept("w", self.setKey, ["w", True])
        self.accept("s", self.setKey, ["s", True])
        self.accept("a", self.setKey, ["a", True])
        self.accept("d", self.setKey, ["d", True])
        self.accept("w-up", self.setKey, ["w", False])
        self.accept("s-up", self.setKey, ["s", False])
        self.accept("a-up", self.setKey, ["a", False])
        self.accept("d-up", self.setKey, ["d", False])

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
        density_per_100_sq_m = 4

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

            # A random position is generated for the tree within the defined area.
            x = random.uniform(-area_side / 2, area_side / 2)
            y = random.uniform(-area_side / 2, area_side / 2)
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
        self.camera.reparentTo(self.render)
        # Set initial camera position at 2m height, behind the drone
        self.camera.setPos(0, 0, 2)
        self.camera.lookAt(self.drone)


app = DroneSimulation()
app.run()
