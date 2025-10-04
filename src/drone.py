from panda3d.core import CardMaker, NodePath


class Drone:

    def __init__(self, length=0.3):
        """Creates a drone procedurally."""
        card_maker = CardMaker("drone")
        card_maker.setFrame(-0.5, 0.5, -0.5, 0.5)

        cube = NodePath("drone")

        # Create the 6 faces of the cube
        faces = [NodePath(card_maker.generate()) for _ in range(6)]

        # front
        faces[0].reparentTo(cube)
        faces[0].setPos(0, -length / 2, 0)

        # back
        faces[1].reparentTo(cube)
        faces[1].setPos(0, length / 2, 0)
        faces[1].setH(180)

        # right
        faces[2].reparentTo(cube)
        faces[2].setPos(length / 2, 0, 0)
        faces[2].setH(-90)

        # left
        faces[3].reparentTo(cube)
        faces[3].setPos(-length / 2, 0, 0)
        faces[3].setH(90)

        # top
        faces[4].reparentTo(cube)
        faces[4].setPos(0, 0, length / 2)
        faces[4].setP(-90)

        # bottom
        faces[5].reparentTo(cube)
        faces[5].setPos(0, 0, -length / 2)
        faces[5].setP(90)

        self.mesh = cube
