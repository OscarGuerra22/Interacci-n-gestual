import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
import cv2
import time
import pygame
import pymunk
# import math
# import re



model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
detection_result = None

tips_id = [4,8,12,16,20]



def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  global detection_result
  detection_result = result


def draw_landmarks_on_image(rgb_image, detection_result):

  hand_landmarks_list = detection_result.hand_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
  

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

#--------------------------------------------------------------------------------------------------------------------------


# Configuración de Pygame
pygame.init()
screen_w = 640
screen_h = 480
screen = pygame.display.set_mode((screen_w, screen_h))
clock = pygame.time.Clock()

background = pygame.image.load("background.jpg")
background = pygame.transform.scale(background, (screen_w, screen_h))

font = pygame.font.Font("fuente.TTF", 26)

text_to_show = ""
time_text_setted = 0
text_pos = 0,0

def SetText(text, pos):
  global text_pos
  global text_to_show, time_text_setted
  text_to_show = text
  time_text_setted = pygame.time.get_ticks()
  text_pos = pos
def RenderText():
  global text_to_show, time_text_setted
  if (text_to_show != ""):
    if ((pygame.time.get_ticks() - time_text_setted) > 1500):
      text_to_show = ""
    text_surface = font.render(text_to_show, True, (81, 246, 224))
    screen.blit(text_surface, text_pos)

def convert_coordinates(point):
  return int(point[0]), screen_h - int(point[1])

def convert_vertices_to_pygame(vertices, body):
    absolute_vertices = [body.local_to_world(vertex) for vertex in vertices]
    return absolute_vertices


# Configuración de Pymunk
space = pymunk.Space()
space.gravity = (0, 0)  # Sin gravedad, porque no queremos que las balas caigan

class KillerObstacle:
  def __init__(self, position1, position2, radius):
    self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    self.positions = [pymunk.Vec2d(*position1), pymunk.Vec2d(*position2)]
    self.body.position = position1
    self.shape = pymunk.Circle(self.body, radius)
    self.shape.color =  (200, 0, 255)
    self.shape.collision_type = 5
    self.active = False
    self.velocity = (self.positions[1] - self.positions[0]).normalized() * 5 
  def Activate(self):
    self.active = True
    space.add(self.body, self.shape)
  def Disable(self):
    self.active = False
    space.remove(self.body, self.shape)
  def is_near(self, position, threshold=5):
    return (self.body.position - position).length < threshold
  def Render(self):
    if self.active:
      if self.is_near(self.positions[0]):
          self.velocity = (self.positions[1] - self.positions[0]).normalized() * 5
      elif self.is_near(self.positions[1]):
          self.velocity = (self.positions[0] - self.positions[1]).normalized() * 5
      self.body.position += self.velocity
      pygame.draw.circle(screen, self.shape.color, (int(self.body.position.x), int(self.body.position.y)), int(self.shape.radius))

class OrbitObstacle:
  def __init__(self, position, radius):
    self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
    self.body.position = position
    self.shape = pymunk.Circle(self.body, radius)
    self.base_img = pygame.image.load("mars.png")
    self.base_img = pygame.transform.scale(self.base_img,(self.shape.radius*2, self.shape.radius*2 ))
    self.shape.collision_type = 4
    self.shape.density = 1
    self.active = False
  def Activate(self):
    self.active = True
    space.add(self.body, self.shape)
  def Disable(self):
    self.active = False
    space.remove(self.body, self.shape)
  def Render(self):
    if (self.active):
      screen.blit(self.base_img, (int(self.body.position.x) - self.shape.radius, int(self.body.position.y) - self.shape.radius))

class StickyObstacle:
  def __init__(self, position, vertices):
    self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
    self.body.position = position
    self.shape = pymunk.Poly(self.body, vertices)
    self.shape.color = (116, 205, 38) # Valores RGB: verde grisaceo
    self.shape.collision_type = 3
    self.shape.density = 1
    self.active = False
  def Activate(self):
    self.active = True
    space.add(self.body, self.shape)
  def Disable(self):
    self.active = False
    space.remove(self.body, self.shape)
  def Render(self):
    if (self.active):
      vertices = convert_vertices_to_pygame(self.shape.get_vertices(), self.body)
      pygame.draw.polygon(screen, self.shape.color, vertices)

class BasicObstacle:
  def __init__(self, position, vertices):
    self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
    self.body.position = position
    self.shape = pymunk.Poly(self.body, vertices)
    self.shape.color = (102, 86, 68) # Valores RGB: Gris - marrón
    self.shape.elasticity = 1
    self.shape.density = 1
    self.active = False
  def Activate(self):
    self.active = True
    space.add(self.body, self.shape)
  def Disable(self):
    self.active = False
    space.remove(self.body, self.shape)
  def Render(self):
    if (self.active):
      vertices = convert_vertices_to_pygame(self.shape.get_vertices(), self.body)
      pygame.draw.polygon(screen, self.shape.color, vertices)

class Objective:
  def __init__(self, position):
    self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
    self.body.position = position
    self.shape = pymunk.Circle(self.body, 15)
    self.base_img = pygame.image.load("diana.png")
    self.base_img = pygame.transform.scale(self.base_img,(self.shape.radius*2, self.shape.radius*2 ))
    self.shape.sensor = True
    self.shape.collision_type = 2
    space.add(self.body, self.shape)
  def Render(self):
    screen.blit(self.base_img, (int(self.body.position.x) - self.shape.radius, int(self.body.position.y) - self.shape.radius))

class Bullet:
  def __init__(self, starting_pos, rotation, id):
    self.body = pymunk.Body()
    self.body.position = starting_pos
    self.shape = pymunk.Circle(self.body, 5)
    self.shape.color = (255, 0, 111) # Valores RGB: Rojo - rosado
    self.shape.density = 1
    self.shape.friction = 1
    self.shape.collision_type = 1
    self.shape.elasticity = 1
    self.shape.id = id
    self.id = id
    space.add(self.body, self.shape)
    impulse = 50000 * pymunk.vec2d.Vec2d(1,0)
    impulse = impulse.rotated(-rotation)
    self.body.apply_impulse_at_world_point(impulse, self.body.position)
  def InsideBounds(self):
    return ((self.body.position.x >= 0) & (self.body.position.x <= screen_w) 
             & (self.body.position.y >= 0) & (self.body.position.y <= screen_h))
  def Render(self):
    pygame.draw.circle(screen,self.shape.color,self.body.position,self.shape.radius)
  def __del__(self):
    space.remove(self.body, self.shape)

# Clase para contener las físicas y la representación de la nave espacial
class SpaceShip:
  def __init__(self):
    self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    self.body.position = (400, 240)  # Posición inicial
    self.shape = pymunk.Circle(self.body, 30)
    space.add(self.body, self.shape)
    self.base_img = pygame.image.load("cohete_espacial.png")
    self.base_img = pygame.transform.scale(self.base_img,(self.shape.radius*2, self.shape.radius*2 ))
    self.base_img = pygame.transform.rotate(self.base_img,-90)
    self.rotated_img = pygame.transform.rotate(self.base_img,-180)
    self.need_to_reload = True
    self.bullets = []
    self.last_id = 0
  def UpdateParams(self, index_finger_mcp, middle_finger_mcp, index_finger_tip, middle_finger_tip):
    # Rotación y disparo        
    index_tip = np.array((index_finger_tip.x, index_finger_tip.y))
    middle_tip = np.array((middle_finger_tip.x, middle_finger_tip.y))
    tips_distance = np.linalg.norm(index_tip - middle_tip)
    if (tips_distance < 0.08):
      self.need_to_reload = False
      # pymunk usa ángulos en radianes, mientras que pygame usa ángulos en grados
      angle_radians = np.arctan2(-(index_finger_tip.y - index_finger_mcp.y), index_finger_tip.x - index_finger_mcp.x)
      angle_degrees = math.degrees(angle_radians)
      # asignamos el ángulo correspondiente al body
      self.body.angle = angle_radians
      # rotamos la imagen para que represente el ángulo de disparo que tiene el body
      self.rotated_img = pygame.transform.rotate(self.base_img, angle_degrees)
    elif ((tips_distance > 0.15) & (self.need_to_reload == False)):
      self.Shoot()
      self.need_to_reload = True
    # Posición
    self.body.position = self.body.position.x , int(((index_finger_mcp.y + middle_finger_mcp.y) / 2) * 480)
  def Render(self):
    screen.blit(self.rotated_img, (int(self.body.position.x) - self.shape.radius, int(self.body.position.y) - self.shape.radius))
    self.bullets = [bullet for bullet in self.bullets if bullet.InsideBounds()]
    for bullet in self.bullets:
        bullet.Render()
  def Shoot(self):
    self.bullets.append(Bullet(self.body.position, self.body.angle, self.last_id))
    self.last_id += 1
  def ClearBullets(self):
    self.bullets.clear()
  def KillBullet(self, id):
    self.bullets = [bullet for bullet in self.bullets if bullet.id != id]

# Creación de objetos
objective = Objective((200, 240))
sship = SpaceShip()
level_1 = []
level_2 = [
  OrbitObstacle((250, 100), 30),
  BasicObstacle((250, 400), [(-100,-20), (100, 20), (-100,20), (100, -20)]),
  BasicObstacle((330, 250), [(-20,-130), (20, 130), (-20, 130), (20, -130)])
]
level_3 = [
  BasicObstacle((250, 240), [(-15,-15), (15, 15), (-15,15), (15, -15)]),
  BasicObstacle((250, 190), [(-100,-20), (100, 20), (-100,20), (100, -20)]),
  StickyObstacle((250, 290), [(-100,-20), (100, 20), (-100,20), (100, -20)])
]
level_4 = [
  KillerObstacle((220,350), (320, 120), 15),
  BasicObstacle((182, 240), [(-1,18), (-1, -20), (1, 18), (1, -20)]),
  BasicObstacle((218, 240), [(-1,18), (-1, -20), (1, 18), (1, -20)]),
  BasicObstacle((200, 259), [(19,-1), (-19, -1), (19, 1), (-19, 1)]),
  BasicObstacle((240, 140), [(0,-20), (20, 20), (-20,25)]),
  BasicObstacle((280, 340), [(0,20), (20, -20), (-20,-25)]),
  BasicObstacle((340, 170), [(0,-20), (20, 20), (-20,25)]),
]
current_level = -1
levels = [level_1, level_2, level_3, level_4]

def NextLevel():
  global current_level
  if current_level >= 0:
    for element in levels[current_level]:
      element.Disable()
  current_level += 1
  for element in levels[current_level]:
    element.Activate()
  
NextLevel()

# Colisiones
links = []
# Funciones
def AdvanceRound(arbiter,space,data):
  if (current_level < (len(levels) - 1)):
    for link in links:
      space.remove(link)  # Elimina cada joint del espacio
    links.clear()
    SetText("Avanzas de ronda", (170, 100))
    NextLevel()
    sship.ClearBullets()
  else: 
    SetText("Has ganado",  (230, 100))
  return False

def StickToObstacle(arbiter, space, data):
    bullet_body = arbiter.shapes[0].body
    obstacle_body = arbiter.shapes[1].body
    joint = pymunk.PinJoint(bullet_body, obstacle_body)
    space.add(joint)
    links.append(joint)
def StopBullet(arbiter, space, data):
  bullet_body = arbiter.shapes[0].body
  bullet_body.velocity = 0,0
  return True

def KillBullet(arbiter, space, data):
  sship.KillBullet(arbiter.shapes[0].id)
  return False

# Handlers
# Bala con objetivo:
bullet_objective_hanlder = space.add_collision_handler(1,2)
bullet_objective_hanlder.begin = AdvanceRound

bullet_sticky_obs_hanlder = space.add_collision_handler(1,3)
bullet_sticky_obs_hanlder.begin = StopBullet
bullet_sticky_obs_hanlder.separate = StickToObstacle

bullet_orbit_handler = space.add_collision_handler(1,4)
bullet_orbit_handler.separate = StickToObstacle

bullet_destroyer_handler = space.add_collision_handler(1,5)
bullet_destroyer_handler.begin = KillBullet

def RenderAll():
  screen.blit(background, (0, 0))
  objective.Render()
  sship.Render()
  for obstacle in levels[current_level]:
    obstacle.Render()
  RenderText()


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
  cap = cv2.VideoCapture(0)
  running = True
  while cap.isOpened() and running:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image,1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, frame_timestamp_ms)
    if detection_result is not None:
      image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      if len(detection_result.hand_landmarks) > 0:
        landmarks = detection_result.hand_landmarks[0]
      
        # Obtener coordenadas de los punto necesarios
        index_finger_mcp = landmarks[5]
        middle_finger_mcp = landmarks[9]
        index_finger_tip = landmarks[8]
        middle_finger_tip = landmarks[12]
        sship.UpdateParams(index_finger_mcp, middle_finger_mcp, index_finger_tip, middle_finger_tip)
        
    # Avanzar la simulación de Pymunk
    space.step(1 / 60.0)
    # Renderizar el objeto en Pygame
    screen.fill((255, 255, 255))
    RenderAll()
    pygame.display.flip()
    clock.tick(60)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
pygame.quit() 