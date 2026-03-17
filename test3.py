import pygame
import math
import heapq
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

cell_size = 60
fps = 60

WHITE = (255, 255, 255)
BLACK = (40, 40, 40)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
BLUE = (52, 152, 219)
GREY = (200, 200, 200)
YELLOW = (241, 196, 15)
LIGHT_GREY = (220, 220, 220)
DARK_GREY = (170, 170, 170)

# rows = int(22)
# columns = int(10)

maze = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

rows = len(maze)
cols = len(maze[0])
width = cols * cell_size
height = rows * cell_size


# A* Algorithm
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def astar(maze, start, end, screen, walls):
    start_node = Node(start)
    end_node = Node(end)
    open_list = []
    closed_set = set()
    open_set_tracker = {start}
    heapq.heappush(open_list, start_node)

    # Draw inital empty maze
    screen.fill(WHITE)
    for wall in walls:
        pygame.draw.rect(screen, BLACK, wall)

    # Draw start and end grid
    pygame.draw.rect(screen, GREEN, (start[1] * cell_size, start[0] * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, RED, (end[1] * cell_size, end[0] * cell_size, cell_size, cell_size))
    pygame.display.flip()

    # wait for user input before starting
    print("Press any key to start A* visualization...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN: waiting = False
            if event.type == pygame.QUIT: sys.exit()

    while open_list:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        current_node = heapq.heappop(open_list)
        open_set_tracker.remove(current_node.position)
        closed_set.add(current_node.position)

        # areas explored highlighted in light gray
        if current_node.position != start and current_node.position != end:
            rect = pygame.Rect(current_node.position[1] * cell_size, current_node.position[0] * cell_size, cell_size,
                               cell_size)
            pygame.draw.rect(screen, LIGHT_GREY, rect)
            pygame.display.update(rect)

        if current_node.position == end_node.position:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in directions:
            node_pos = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])

            if (node_pos[0] > (rows - 1) or node_pos[0] < 0 or node_pos[1] > (cols - 1) or node_pos[1] < 0): continue
            if maze[node_pos[0]][node_pos[1]] != 0: continue
            if node_pos in closed_set: continue

            new_node = Node(node_pos, current_node)
            new_node.g = current_node.g + 1
            new_node.h = abs(new_node.position[0] - end_node.position[0]) + abs(
                new_node.position[1] - end_node.position[1])
            new_node.f = new_node.g + new_node.h

            if node_pos not in open_set_tracker:
                heapq.heappush(open_list, new_node)
                open_set_tracker.add(node_pos)

                # areas newly explored is in dark gray
                if node_pos != start and node_pos != end:
                    rect = pygame.Rect(node_pos[1] * cell_size, node_pos[0] * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, DARK_GREY, rect)
                    pygame.display.update(rect)

        pygame.time.delay(30)

    return []


def get_closest_point_on_segment(A, B, P):
    ab_x, ab_y = B[0] - A[0], B[1] - A[1]
    ap_x, ap_y = P[0] - A[0], P[1] - A[1]

    ab_squared = ab_x ** 2 + ab_y ** 2
    if ab_squared == 0: return A

    t = (ap_x * ab_x + ap_y * ab_y) / ab_squared
    t = max(0.0, min(1.0, t))

    return (A[0] + t * ab_x, A[1] + t * ab_y)


walls = []
for row in range(rows):
    for col in range(cols):
        if maze[row][col] == 1:
            walls.append(pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size))


class HapticRobot:
    def __init__(self, start_pos):
        self.pos = list(start_pos)
        self.vel = [0.0, 0.0]
        self.radius = 12
        self.mass = 1.0
        self.damping = 0.85
        self.max_velocity = 300.0

        self.user_force = [0.0, 0.0]
        self.reaction_force = [0.0, 0.0]

        # settings, change here to alter pull towards A* path
        self.guideline_stiffness = 25.0
        self.tunnel_radius = 10.0

    def apply_forces(self, force_x, force_y, astar_waypoints, dt):
        self.user_force = [force_x, force_y]
        self.reaction_force = [0.0, 0.0]
        fx, fy = force_x, force_y

        # Virtual Fixture
        if len(astar_waypoints) > 1:
            closest_dist = float('inf')
            closest_target = self.pos

            # find the closest point along the A* path
            for i in range(len(astar_waypoints) - 1):
                A = astar_waypoints[i]
                B = astar_waypoints[i + 1]
                target = get_closest_point_on_segment(A, B, self.pos)

                dist = math.hypot(target[0] - self.pos[0], target[1] - self.pos[1])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_target = target
            # if the robot moves more than the determined distance away from a* then pull it back
            if closest_dist > self.tunnel_radius:
                pull_dx = closest_target[0] - self.pos[0]
                pull_dy = closest_target[1] - self.pos[1]

                guide_fx = pull_dx * self.guideline_stiffness
                guide_fy = pull_dy * self.guideline_stiffness

                fx += guide_fx
                fy += guide_fy

                self.reaction_force[0] += guide_fx
                self.reaction_force[1] += guide_fy

        ax = fx / self.mass
        ay = fy / self.mass
        self.vel[0] = (self.vel[0] + ax * dt) * self.damping
        self.vel[1] = (self.vel[1] + ay * dt) * self.damping

        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

    def resolve_wall_collisions(self, walls):
        for wall in walls:
            closest_x = max(wall.left, min(self.pos[0], wall.right))
            closest_y = max(wall.top, min(self.pos[1], wall.bottom))
            dx, dy = self.pos[0] - closest_x, self.pos[1] - closest_y
            distance = math.hypot(dx, dy)

            if distance < self.radius:
                overlap = self.radius - distance
                if distance == 0: dx, dy, distance = 1, 0, 1
                normal_x, normal_y = dx / distance, dy / distance

                self.pos[0] += normal_x * overlap
                self.pos[1] += normal_y * overlap

                dot_product = self.vel[0] * normal_x + self.vel[1] * normal_y
                if dot_product < 0:
                    self.vel[0] -= dot_product * normal_x
                    self.vel[1] -= dot_product * normal_y

                user_push = self.user_force[0] * -normal_x + self.user_force[1] * -normal_y
                if user_push > 0:
                    self.reaction_force[0] += normal_x * user_push
                    self.reaction_force[1] += normal_y * user_push


class MazeControllerNode(Node):
    def __init__(self):
        super().__init__('maze_controller_node')

        pygame.init()
        start_grid = (0, 0)
        end_grid = (8, 11)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ROS 2 Haptic Maze Node")


        start_pixel = (start_grid[1] * cell_size + cell_size // 2,
                       start_grid[0] * cell_size + cell_size // 2)

        self.robot = HapticRobot(start_pos=start_pixel)
        self.walls = walls

        # Get grid path from A*
        grid_path = astar(maze, start_grid, end_grid, self.screen, walls)

        self.pixel_waypoints = []
        for step in grid_path:
            px = step[1] * cell_size + cell_size // 2
            py = step[0] * cell_size + cell_size // 2
            self.pixel_waypoints.append((px, py))

        # Subscribers

        self.force_sub = self.create_subscription(
            Float32MultiArray, '/joint_state', self.force_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Float32MultiArray, '/cmd_vel', 10)
        self.feedback_pub = self.create_publisher(Float32MultiArray, '/feedback', 10)

        self.timer = self.create_timer(1.0 / fps, self.physics_loop)

        self.current_user_force_x = 0.0
        self.current_user_force_y = 0.0

    def force_callback(self, msg):
        if len(msg.data) >= 2:
            self.current_user_force_x = msg.data[0]
            self.current_user_force_y = msg.data[1]

    def physics_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        dt = 1.0 / fps
        self.robot.apply_forces(self.current_user_force_x, self.current_user_force_y, self.pixel_waypoints, dt)
        self.robot.resolve_wall_collisions(self.walls)

        vel_msg = Float32MultiArray()
        vel_msg.data = [float(self.robot.vel[0]), float(self.robot.vel[1])]
        self.cmd_vel_pub.publish(vel_msg)

        feedback_msg = Float32MultiArray()
        feedback_msg.data = [float(self.robot.reaction_force[0]), float(self.robot.reaction_force[1])]
        self.feedback_pub.publish(feedback_msg)

        # Drawing the GUI
        self.screen.fill(WHITE)

        if len(self.pixel_waypoints) > 1:
            pygame.draw.lines(self.screen, YELLOW, False, self.pixel_waypoints, 4)

        for wall in self.walls:
            pygame.draw.rect(self.screen, BLACK, wall)

        pygame.draw.circle(self.screen, BLUE, (int(self.robot.pos[0]), int(self.robot.pos[1])), self.robot.radius)

        pygame.display.flip()