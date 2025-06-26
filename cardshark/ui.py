"""Provides tools to create a graphical user interface for your games

"""

# python stuff
from abc import ABC, abstractmethod
import typing

# UI stuff
import pygame
import pygame_gui

class UIStateBase(ABC):
    """Represents a state of the UI e.g. "main menu" or "player turn" etc.

    Your UI state classes should implent the draw() and handle_event() methods. 

    **draw()**:

    should render the current frame of the game to the window_sirface attribute

    e.g.

    .. code::

        def draw(self):
        
            background = pygame.Surface(self.manager.window_shape)

            background.fill(pygame.Color('green'))

            self.window_surface.blit(background, (0, 0))

    .. endcode::

    will render a solid green field


    **handle_event()**

    should, as the name implies, handle events_ emmitted by pygame. 
    Should handle the passed event, e.g. button press to make some move, and return
    a state object indicating the new UI state (can return itself if the state doesn't change).
    Should also return a boolean indicating whether the event was successfully handled. If true,
    will move to the new state, if false, nothing will change. 

    .. _events: https://pygame-gui.readthedocs.io/en/latest/events.html#gui-events

    If you have two UI states - a main menu state, UIStateMainMenu, and a game state UIGameState - 
    then a simple example might look like:

    .. code::

        class UIStartMenuState(UIStateBase):
        
        ...

            def handle_event(self, event:pygame.event.Event):

                ## most ui events will probably be button presses
                if event.type == pygame_gui.UI_BUTTON_PRESSED:

                    ## the start game button was pressed so we return the game state
                    ## and tell the manager that the event was handled successfuly by 
                    ## also returning true
                    if event.ui_element == self.start_game_button:
                        print('Starting game!')
                        return UIGameState, True
                
                ## otherwise we tell the manager that the event was something else 
                ## that we don't know about so we didn't handle it                
                return UIStstartMenuState, False

    .. endcode::

    """

    instance = None

    def __init__(self, manager:'UIManagerBase'):
        self.ui_manager = manager
        self.window_surface = manager.window_surface

    @classmethod
    def get(cls, manager:'UIManagerBase'):
        if cls.instance is None:
            print(f"setting new instance for {cls.__name__}")
            cls.instance = cls(manager)        
    
        return cls.instance
        
    @staticmethod
    @abstractmethod  
    def draw(self, manager:'UIManagerBase'):
        """Draw the UI for this state: buttons, game board etc."""

    @staticmethod
    @abstractmethod  
    def handle_event(self, event:pygame.event.Event) -> typing.Tuple['UIStateBase', bool]:
        """Handle the possible events that can be generated in this state.
        
        Should return true if the event was handled correctly, false otherwise"""

    def quit(self):
        """Request the manager to quit the application."""

        pygame.event.post(pygame.event.Event(pygame.QUIT))

    def hide(self):
        ## Hides all ui elements associated with this state
        ## to be used by the manager when switching between states
        ## *should not* be used anywhere else

        for attr in [self.__dict__[a] for a in self.__dict__.keys() if not a.startswith('_')]:
            if isinstance(attr, pygame_gui.core.ui_element.UIElement):
                print(f"hiding {attr}")
                attr.hide()

    def show(self):
        ## Shows all ui elements associated with this state
        ## to be used by the manager when switching between states
        ## *should not* be used anywhere else
        
        for attr in [self.__dict__[a] for a in self.__dict__.keys() if not a.startswith('_')]:
            if isinstance(attr, pygame_gui.core.ui_element.UIElement):
                print(f"showing {attr}")
                attr.show()



class UIManagerBase(ABC):

    def __init__(self, shape:tuple[int], theme_path:str, name:str="Game"):

        pygame.init()

        pygame.display.set_caption(name)

        self.window_shape = shape

        self.pgui_manager = pygame_gui.UIManager(shape, theme_path=theme_path)
        self.window_surface = pygame.display.set_mode(shape)

        self.clock = pygame.time.Clock()

        self.is_running = True

        self.current_state = None

    
    def get_pgui_manager(self):

        return self.pgui_manager
    

    def run(self):
        
        while self.is_running:

            time_delta = self.clock.tick(60)/1000.0

            for event in pygame.event.get():

                new_state, handled = self.current_state.handle_event(event)
                if(handled):
                    if not isinstance(self.current_state, new_state):
                        self.current_state.hide()
                        self.current_state = new_state.get(self)
                        self.current_state.show()

                    continue

                if event.type == pygame.QUIT:

                    self.is_running = False

                self.pgui_manager.process_events(event)


            self.pgui_manager.update(time_delta)

            self.current_state.draw()

            self.pgui_manager.draw_ui(self.current_state.window_surface)

            pygame.display.update()

