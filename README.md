## Fallando hasta el Éxito con Hydra

Este es el código que utilicé como base en la demostración de la Charla en la Pycon2021. Siéntete libre de utilizarlo y experimentar con él a tu antojo, ojalá te sirva!! 🤘

### Ejecución simple

El código se puede ejecutar de la siguiente manera en Modo Simple:

```python
python main.py +preprocess=simple +models=lr
```

```python
python main.py +preprocess=simple +models=rf
```

```python
python main.py +preprocess=complex +models=lr
```

```python
python main.py +preprocess=complex +models=rf
```

Cualquiera de estos comandos ejecutará un Modelo Simple con una configuración de Preprocesamiento y de Modelo fija. 

Recordar que utilizando el operador `++` se puede hacer override e ir variando la configuración, por ejemplo:

```python
python main.py +preprocess=complex +models=lr ++models/C=0.1
```
### Ejecución con Búsqueda de Hiperparámetros

En el caso de querer utilizar Optuna para correr multiples modelos con búsqueda de Hiperparámetros utilizar lo siguiente:

```python
python main-multirun.py +preprocess=multi_simple +models=multi_lr
```

```python
python main-multirun.py +preprocess=multi_simple +models=multi_rf
```

```python
python main-multirun.py +preprocess=multi_complex +models=multi_lr
```

```python
python main-multirun.py +preprocess=multi_complex +models=multi_rf
```

Recordar también que es posible utilizar el Flag `-m` para ejecutar multiples ejecuciones:

```python
main-multirun.py -m +preprocess=multi_simple,multi_complex +models=multi_rf,multi_lr
```

En este caso se ejecutará **Random Forest y Regresión Logística** con preprocesamiento Complex.

> Nota: Es posible correr `Optuna` en formato Multithread. Existe la opción `n_jobs`, igual que en `Scikit-Learn`. Si te interesa utilizarla puedes agregarla en `.optimize()` pero pronto será deprecada por Problemas con el GIL por lo cual no quise incluirla. Si quieres apurar la búsqueda puedes modificar el código a conveniencia.

