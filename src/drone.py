from panda3d.core import Vec3, CardMaker, TransparencyAttrib
import random


class Drone:

    def __init__(self, loader, render):
        """Creates a drone from an image."""
        # The drone is a NodePath, the top-level object for the drone in the scene.
        self.node = render.attachNewNode("drone")

        # Load the drone texture.
        drone_texture = loader.loadTexture("assets/images/drone.png")

        # Create a card (a flat rectangle) to display the texture.
        cm = CardMaker("drone-card")
        # The card is created with a size of 1x1.
        # We'll set the frame to be centered at (0,0,0)
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        self.mesh = self.node.attachNewNode(cm.generate())

        # Apply the texture to the card.
        self.mesh.setTexture(drone_texture)

        # Enable transparency to see through the PNG's alpha channel.
        self.mesh.setTransparency(TransparencyAttrib.M_alpha)

        # The user requested a 40cm width, which is 0.4 meters.
        # We need to scale the card to match the aspect ratio of the texture.
        tex_width = drone_texture.getXSize()
        tex_height = drone_texture.getYSize()
        aspect_ratio = tex_height / tex_width
        drone_height = 0.4 * aspect_ratio
        self.node.setScale(0.4, 1, drone_height)

        # Make the drone always face the camera.
        self.node.setBillboardPointEye()

        # Set initial position 2m above the ground
        self.node.setPos(random.uniform(-20, 20), random.uniform(50, 80), 10)

        # The speed of the drone in meters per second.
        self.speed = 50000.0
        # The intensity of the random Brownian motion as a factor of speed.
        self.brownian_factor = 0.05  # 10% of speed
        # Controls how smoothly the Brownian motion changes direction.
        # A higher value results in slower, smoother, more drifting changes.
        self.brownian_smoothness = 0.5
        # Stores the current direction of the Brownian motion vector.
        self.brownian_direction = Vec3(0, 0, 0)

    def update(self, camera, dt):
        """
        Moves the drone towards the camera in a straight line with Brownian motion.
        """
        direction_vec = camera.getPos() - self.node.getPos()
        distance_to_camera = direction_vec.length()

        if distance_to_camera > 0.5:  # 50cm threshold
            direction_vec.normalize()

            # The distance the drone moves in this frame, clamped to not overshoot the target.
            distance_this_frame = min(self.speed * dt, distance_to_camera)
            linear_movement = direction_vec * distance_this_frame

            # To create the Brownian motion, we need two vectors that are perpendicular to the direction vector.
            up_vec = Vec3(0, 0, 1)
            if abs(direction_vec.dot(up_vec)) > 0.999:
                perp_vec1 = direction_vec.cross(Vec3(1, 0, 0))
            else:
                perp_vec1 = direction_vec.cross(up_vec)
            perp_vec1.normalize()
            perp_vec2 = direction_vec.cross(perp_vec1)
            perp_vec2.normalize()

            # --- Smoothed Brownian Motion ---
            # Generate a new random target direction in the perpendicular plane.
            random_x = random.uniform(-1.0, 1.0)
            random_y = random.uniform(-1.0, 1.0)
            target_brownian_vec = (perp_vec1 * random_x + perp_vec2 * random_y)

            # We use a simple linear interpolation (lerp) to smoothly transition the
            # current Brownian direction towards the new random target.
            # This makes the motion fluid rather than jerky.
            smoothing = 1.0 / self.brownian_smoothness
            lerp_alpha = dt / (dt + smoothing)

            # If this is the first frame, jump directly to the target vector.
            if self.brownian_direction.length_squared() == 0:
                self.brownian_direction = target_brownian_vec
            else:
                self.brownian_direction = (
                    self.brownian_direction * (1 - lerp_alpha)
                    + target_brownian_vec * lerp_alpha
                )

            self.brownian_direction.normalize()

            # The final Brownian motion offset for this frame.
            # It is scaled proportionally to the distance the drone moves in this frame.
            brownian_offset = (
                self.brownian_direction * distance_this_frame * self.brownian_factor
            )

            # The new position is the linear path plus the Brownian motion.
            new_pos = self.node.getPos() + linear_movement + brownian_offset
            self.node.setPos(new_pos)
