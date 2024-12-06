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

font = pygame.font.Font(None, 36)

text_to_show = ""
time_text_setted = 0

def SetText(text):
  global text_to_show, time_text_setted
  text_to_show = text
  time_text_setted = pygame.time.get_ticks()

def RenderText():
  global text_to_show, time_text_setted
  if (text_to_show != ""):
    if ((pygame.time.get_ticks() - time_text_setted) > 1500):
      text_to_show = ""
    text_surface = font.render(text_to_show, True, (0, 0, 0))
    screen.blit(text_surface, (200, 100))

def convert_coordinates(point):
  return int(point[0]), screen_h - int(point[1])

# Configuración de Pymunk
space = pymunk.Space()
space.gravity = (0, 0)  # Sin gravedad, porque no queremos que las balas caigan

class Objective:
  def __init__(self, position):
    self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
    self.body.position = position
    self.shape = pymunk.Circle(self.body, 15)
    self.shape.color = pygame.Color("yellow")
    self.shape.sensor = True
    self.shape.collision_type = 2
    space.add(self.body, self.shape)
  def Render(self):
    pygame.draw.circle(screen,self.shape.color,self.body.position,self.shape.radius)

class Bullet:
  def __init__(self, starting_pos, rotation):
    self.body = pymunk.Body()
    self.body.position = starting_pos
    self.shape = pymunk.Circle(self.body, 5)
    self.shape.color = pygame.Color("red")
    self.shape.density = 1
    self.shape.friction = 0.1
    self.shape.collision_type = 1
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
    self.shape = pymunk.Circle(self.body, 20)
    space.add(self.body, self.shape)
    self.base_img = pygame.image.load("spaceship.png")
    self.base_img = pygame.transform.scale(self.base_img,(self.shape.radius*2, self.shape.radius*2 ))
    self.base_img = pygame.transform.rotate(self.base_img,-90)
    self.rotated_img = self.base_img
    self.need_to_reload = True
    self.bullets = []
  def UpdatePositionAndRotation(self, index_finger_mcp, middle_finger_mcp, index_finger_tip, middle_finger_tip):
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
    self.bullets.append(Bullet(self.body.position, self.body.angle))

# Colisiones
# Funciones
def AdvanceRound(arbiter,space,data):
  SetText("Avanzas de ronda")
  return False

# Handlers
# Bala con objetivo:
bullet_objective_hanlder = space.add_collision_handler(1,2)
bullet_objective_hanlder.begin = AdvanceRound

# Creación de objetos
objective = Objective((200, 240))
sship = SpaceShip()

def RenderAll():
  objective.Render()
  sship.Render()
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
        sship.UpdatePositionAndRotation(index_finger_mcp, middle_finger_mcp, index_finger_tip, middle_finger_tip)
        
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