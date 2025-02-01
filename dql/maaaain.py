
        ## Si el cuadrito tiene verde, me debe dar una ponderacion de accion hacia ql
        ## Si el cuadrito tiene negro, ponderacion de accion hacia emp
        #  Si hay cuadrito verde, la probabilidad de seguir ql y tener exito es alta, pero la de seguir emp y tener exito es baja
        # POLICY GRADIENTS
        # MAXIMUM LIKELIHOOD en vez de aproximar expected values
        # Maximum likelihood -> predicción segun lo que ya llevas de resultados, para aproximar.
        # SOFTMAX-> derivarla, -prob de a * la de b
        # parametrizar una softmax, tal que la prob de A sea tanto y la de B sea el resto. Utilizando max lik
        ## valor esperado del gradiente del logaritmo de la probabilidad de la accion tomada multiplicada por el reward [prom-reward encontrdo    
        # ,  reward del valor esperado] que encontró
        ## si hay cuadrito verde y siguio a ql, entonces reward positivo. Si no hay verde y elige emp, reward positivo, al reves reward negativo


        ## pytorch, keras. PG
        ## aproximamos probabilidades, y esas son la ponderaciones por la cuales multiplicariamos las q tables.
