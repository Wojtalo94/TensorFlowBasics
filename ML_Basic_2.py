# 2. XXX z wykorzystaniem zbioru danych Fashion MNIST (70000 zdjęć w 10 różnych kategioriach, zdjęcia mają po 28x28pix).

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
'''
Zbiór danych Fashion MNIST jest wbudowany w TensorFlow, więc można go łatwo załądować za pomocą 
'fashion_mnist.load_data()'
Obrazów szkoleniowych mamy 60000. Pozostałe 10000 to zbiór testowy, który możemy wykorzystać do psrawdzenia, jak dobrze
działa nasza sieć neuronowa.
'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
Zobaczmy jak wyglądają treningowe wartości

'''
plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

'''
Przy projektowaniu sieci neuronowej zawsze warto przyjrzeć się wartościom wejściowym i wyjściowym. Tutaj możemy 
zobaczyć, że nasza sieć neuronowa jest nieco bardziej złożona, niż ta wcześniej. Nasza pierwsza warstwa ma wejście w 
kształcie 28x28 co jest rozmiarem naszego obrazu. Ostatnia warstwa ma numer 10, co odpowiada liczbie różnych elemntów
ubioru reprezentowanych w naszym zestawie danych. Nasza sieć neuronowa będzie więc działać na zasadzie filtra, który 
pobiera zestaw pikseli o wymiarach 28x28 i zwraca 1 z 10 wartości. Natomiast wartość 128 to liczba funkcji, a każda
z nich będzie miała parametry (nazwijmy je od f0 do f127). Chcemy aby po wprowadzeniu do nich pikseli buta, jeden po 
drugim, kombinacja wszystkich tych funkcji dawała poprawną wartość. Aby to zrobić, komputer będzie musiał ustalić 
parametry wewnątrz tych funkcji, aby uzyskać ten wynik. Następnie zostanie to rozszerzone na wszystkie elementy ubioru 
w zbiorze danych. Logika jest taka, że skoro już to zrobił, powinien być w stanie rozpoznawać elementy ubioru. 

Pierwsza z funkcji activation znajduje się w warstwie 128 funkcji i nazywa się 'relu'. Czyli jednostka liniowa
wyprostowana:
if (x>0){
    return x;
    }
    else{
        :return 0;
    }
Tak naprawdę działanie tego mechanizmu jest tak proste, że zwraca wartość, jeżeli jest ona większa od zera. Jeśli więc
funkcja ta ma na wyjściu wartość zero lub mniejszą, jest ona po prostu filtrowana.

Druga z funkcji activation nazywa się 'softmax' i efektem jej działanie jest wybranie największej liczby w zestawie. 
Warstwa wyjściowa tej sieci neuronowej zawiera 10 elementów, które reprezentują prawdopodobieństwo, że patrzymy na ten 
konkretny element ubioru. Więc zamiast przeszukiwać w celu znalezienia najwięszkego, 'softmax' ustawia wartość 1, a 
reszta to 0, więc wszystko, co musimy zrobić, to znaleźć 1.

Sequential: Określa Sekwencję warstw w sieci neuronowej.
Flatten: Pamiętasz, jak wcześniej nasze obrazy były kwadratem, kiedy je wydrukowałeś? Flatten po prostu bierze ten kwadrat i zamienia go w 1-wymiarowy zestaw.
Dense: Dodaje warstwę neuronów
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
Mamy funkcję optimizer i funkcję loss. Sieć neuronowa zostanie zainicjowana losowymi wartościami. Funkcja loss zmierzy
następnie, jak dobre lub złe były wyniki, a następnie za pomocą funkcji optymizera, wygeneruje nowe parametry dla
funkcji, aby sprawdzić, czy da się ją ulepszyć. 
'''
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
Następnie szkolenie jest bardzo proste. Dopasowujemy obrazy szkoleniowe do etykiet szkoleniowych. Tym razem spróbujemy 
tylko dla 5 epok.  
'''
model.fit(train_images, train_labels, epochs=5)
'''
Po zakończeniu szkolenia - powinieneś zobaczyć wartość dokładności na końcu ostatniej epoki. Może ona wyglądać jak 
0,8036. Oznacza to, że sieć neuronowa ma około 80% dokładności w klasyfikowaniu danych treningowych. Oznacza to, że 
dopasowanie wzorca między obrazem a etykietami zadziałało w 80% przypadków. Nie jest to świetny wynik, ale nie jest też 
zły, biorąc pod uwagę, że sieć była trenowana tylko przez 5 epok i została wykonana dość szybko.
'''

'''
Testowe obrazy 'test_images' to obrazy, których model wcześniej nie widział, możemy więć ich użyć do sprawdzenia, jak 
dobrze działa nasz model. Możemy wykonać ten test przekazując je do metody 'evaluate' w ten sposób. 
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
'''
Dla mnie zwróciło to dokładność około .7819, co oznacza, że było to około 78% dokładności. Zgodnie z oczekiwaniami, 
prawdopodobnie nie poradziłby sobie tak dobrze z niewidocznymi danymi, jak z danymi, na których został przeszkolony!
'''
'''
I na koniec możemy uzyskać progrnozy dla nowych obrazów, wywołując 'model.predict' w ten sposób. 
'''
predictions = model.predict(test_images)

print(predictions[0])
print(test_labels[0])

'''
Wynikiem modelu jest lista 10 liczb. Liczby te są prawdopodobieństwem, że klasyfikowana wartość jest odpowiednią 
wartością, tj. pierwsza wartość na liście to prawdopodobieństwo, że pismo odręczne ma wartość „0”, następna to „1” itd. 
Zauważ, że wszystkie te prawdopodobieństwa są BARDZO NISKIE.

W przypadku 7 prawdopodobieństwo wyniosło .999+, czyli sieć neuronowa mówi nam, że jest to prawie na pewno 7.

Skąd wiesz, że ta lista mówi ci, że przedmiot jest butem do kostki?
Nie ma wystarczających informacji, aby odpowiedzieć na to pytanie
Dziesiąty element na liście jest największy, a but za kostkę ma etykietę 9.
But za kostkę ma etykietę 9, a na liście jest 0->9 elementów.
Odpowiedź
Prawidłowa odpowiedź to (2). Zarówno lista, jak i etykiety są oparte na 0, więc but na kostce z etykietą 9 oznacza, że 
jest 10. z 10 klas. Lista, na której 10. element ma najwyższą wartość, oznacza, że sieć neuronowa przewidziała, że 
klasyfikowany przez nią przedmiot to najprawdopodobniej but do kostki
'''
