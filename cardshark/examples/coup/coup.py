# Python stuff
import typing

# cardshark stuff
from cardshark import logging as log

from cardshark.examples.coup import coup_engine, coup_agent
from cardshark.ui import UIManager, UIStateBase

import pygame
import pygame_gui

class CoupUIGameState(UIStateBase):

    def __init__(self, manager:'UIManager'):
        super().__init__(manager)

        self.main_menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((350, 275), (100, 50)),
            text='Return to Main Menu',
            manager=self.ui_manager.get_pgui_manager()
        )

    def draw(self):

        background = pygame.Surface((800, 600))

        background.fill(pygame.Color('green'))

        self.window_surface.blit(background, (0, 0))


    def handle_event(self, event:pygame.event.Event):

        if event.type == pygame_gui.UI_BUTTON_PRESSED:

            if event.ui_element == self.main_menu_button:
                print('Returning to main menu!')
                return CoupUIStstartMenuState, True
            
        return CoupUIGameState, False

class CoupUIStstartMenuState(UIStateBase):

    def __init__(self, manager:'UIManager'):
        super().__init__(manager)

        self.start_game_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((350, 275), (100, 50)),
            text='Start Game',
            manager=self.ui_manager.get_pgui_manager()
        )

        self.quit_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((350, 175), (100, 50)),
            text= 'Quit',
            manager=self.ui_manager.get_pgui_manager()
        )

    def draw(self):

        background = pygame.Surface((800, 600))

        background.fill(pygame.Color('#000000'))

        self.window_surface.blit(background, (0, 0))



    def handle_event(self, event:pygame.event.Event):

        if event.type == pygame_gui.UI_BUTTON_PRESSED:

            if event.ui_element == self.start_game_button:
                print('Starting game!')
                return CoupUIGameState, True

            if event.ui_element == self.quit_button:
                print('Quitting!')
                self.quit() 
                return CoupUIStstartMenuState, True
            
        return CoupUIStstartMenuState, False


ui_man = UIManager(
    CoupUIStstartMenuState,
    name="coup",
    shape = (800, 600),
    theme_path="coup_ui_theme.json"
)
ui_man.run()