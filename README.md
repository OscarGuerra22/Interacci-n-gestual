# Juego SIPC
Juego creado con pygame, pymunk y mediapipe, en el que mediante interacción gestual se debe acertar en un objetivo para avanzar de nivel

## To-do
- **Obstáculos:** Crear clases y definir sus coliones para que interactuen con los disparos del usuario. Ejemplos de obstáculos:
    - Obstáculo normal: no hay ningún tipo de colisión, las bolas rebotan sin más
    - Obstáculo pegajoso: las balas se quedan pegados en el. Se crea mediante una colisión que crea un link
    - Obstáculo destructible: una colisión hace que este desaparezca
    - Otros obstáculos: deberíamos buscar algún otro tipo de link que poder aplicar, para que no sea tan simple. [Ejemplos de links](https://www.pymunk.org/en/latest/pymunk.constraints.html)

- **Añadir imágenes**: las balas y el objetivo ahora mismo son figuras geométricas, deberíamos buscar imágenes que superponerles para que el resultado sea mejor
- **Definir niveles**: Crear una lista de objetos de las clases obstáculo para cada nivel. Unir esas tres listas en una y crear un contador de nivel. En RenderAll, especificar que se rendericen los objetos de la sublista apuntada por el nivel. En la colisión AvanzarNivel poner que se aumente el nivel hasta llegar al 3.
