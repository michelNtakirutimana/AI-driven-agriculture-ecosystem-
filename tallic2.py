from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout

class SmartFarmerApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        self.lbl = Label(text="Muraho Michel! Twandike umusaruro?")
        btn = Button(text="Kanda hano ubare", size_hint=(1, 0.5))
        
        # Iyo ukanze kuri button, ikora ikintu
        btn.bind(on_press=self.hindura_izina)
        
        layout.add_widget(self.lbl)
        layout.add_widget(btn)
        return layout

    def hindura_izina(self, instance):
        self.lbl.text = "Umuhinzi Smart: Umusaruro wanditswe!"

if __name__ == '__main__':
    SmartFarmerApp().run()
