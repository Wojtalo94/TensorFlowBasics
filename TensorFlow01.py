# 1. What the set of numbers X and Y has in common?
from tensorflow import keras
import numpy as np

'''
definicja samego modelu (wytrenowanej sieci neuronowej), w tym przypadku jest pojedynczą warstwą wskazywaną przez
'keras.layers.Dense', a ta warstwa ma w sobie pojedynczy neuron, oznaczony jako 'units=1'. Wprowadzamy również
pojedynczą wartość do sieci neuronowej, która jest wartością 'X', a sieć neuronowa przewiduje, jaka będzie wartość
'Y' dla tej wartości 'X'. Dlatego 'input_shape' to jedna wartość. 
'''
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

'''
podczas kompilacji modelu występują dwie wartości, 'optimizer' i 'loss'. Oto klucz do uczenia maszynowego. Zasada 
działania uczenia maszynowego polega na tym, że model próbuje zgadnąć związek między liczbami. Na przykład może 
przypuszczać że Y=5X+5. A podczas treningu program obliczy, jak dobry lub zły jest ten strzał, korzystając z funkcji
'loss'. Następnie użyje funkcji 'optimizer', aby wygenerować kolejne przypuszczenie. Logika jest taka, że połączenie
tych dwóch funkcji, powoli będzie nas zbliżać do poprawnego wzoru.
'''
model.compile(optimizer='sgd', loss='mean_squared_error')

'''
Dane same w sobie są skonfigurowane jako tablica 'X' i 'Y', a nasz proces dopasowywania ich do siebie mieści się w 
metodzie dopasowania modelu. Dosłowanie mówimy: "dopasuj X do Y i spróbuj 500 razy". 
'''
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

'''
W tym przypadku pętla ta zostanie powtórzona 500 razy, a następnie program zgaduje, oblicza jak dokładne jest to 
zgadywanie, po czym używa funkcji 'optimizer', aby udoskonalić to zgadywanie i tak dalej. 
'''
model.fit(xs, ys, epochs=500)

'''
Szkolimy nasz model. Co się stanie gdy spróbujemy przewidzeć Y, gdy X jest równe 10? 
Y = 2X - 1
Y = 20 - 1
Y = 19
Niestety widzimy że odpowiedź to nie jest 19, lecz coś w rodzaju 18.99988. Jest blisko 19, ale to nie jest jeszcze to. 
Dlaczego? Sieci neuronowe radzą sobie z prawdopodobieństwem, więc obliczyły, że istnieje bardzo duże prawdopodobieństwo,
że związki X i Y to Y=2x-1. Nie ma jednak pewności, że obejmuje ona tylko sześć punktów danych. Wynik jest zbliżony do 
19, ale nie dokładnie wynosi on 19.
'''
print(model.predict([10.0]))
