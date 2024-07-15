- Zu viel Noise notwendig, um Lücken zu füllen -> Residuum = Noise dann
- Daher Datensätze (+, -) gleich mit Abweichung vom Mittelwert generieren
- Trainiert OK mit std ca. gleich dem Wert der für Lückenfüllung notwendig ist. Wie kommt man an eine sinnvolle std?
- Berechnung der "Verschiebungsfaktoren" analog zu 3PINN schon möglich, aber zu schlechte Statistik wegen zu wenig Daten. Wass wenn ich mehr +/- Daten generiere, aber nur auf einem Satz trainiere und den Rest zum Verschieben verwende? -> Wäre mathematisch auf jeden Fall ein bisschen was anderes als 3PINN.


- Am 3PINN-ähnlichsten wäre es
    - +/- std für c zu lernen via Residuen und irgendeinen NN (nicht FINN). Frage ist nur, wie generieren wir Residuendatensatz?
    - Dann für ein Quantil die c+/c- Felder mit c_mean +/- fac * c_std
    - Dann R+ für dieses Feld via Training von FINN auf c+ bestimmen


Ganz andere Idee:
- R ist unabhängig von t
- Wir lernen mehrmals R via FINN aber für verschiedene (kleinere) Zeitintervalle
- Dann noch einmal FINN mit allen t was R_mean ergibt?
- Die Menge an R Daten dann 1zu1 wie in 3PINN verwenden

Oder:
Noise auf c_train (oder mit Gauß-Prozess) -> Mehrmals FINN trainieren -> Viele R -> 3PINN auf R (Ist aber eher Messunsicherheit statt methodischer Unsicherheit wie oben)

-----


Nicht auf vollem Feld trainieren für +/- FINN


----

Milestone Präsentationsterminvorschläge:
20.08