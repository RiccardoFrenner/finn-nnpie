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


----

# 18.07.24 Meeting Redepunkte

## FINN mit gelernten Residuen
- Gelernte Residuen sind OK (Ich könnte noch mehr overfitten, habe das aber für nicht sinnvoll betrachtet)
- Aber man sieht kaum einen Unterschied in den R-Kurven, weil Residuum so klein?
- Überschneiden tun sie sich auch und es wäre ein sehr großer Offset

## R-Kurven für größer werdende Zeitintervalle
- Keine wirkliche Konvergenz zu "richtiger" Kurve erkennbar
- Ab einer bestimmten größe des Zeitintervalls wird auch etwas ganz anderes als freundlich-ähnliches gelernt. Liegt vielleicht an Hyperparametern oder weil das c-Feld da nicht mehr genug Informationen enthält. Da müsste man sich nochmal genau den Loss und das gelernte Feld und den Fehler anschauen.

## Konstante, bewegte Zeitintervalle
- ähnlich wie oben: ab einem gewissen Zeitintervall wird nicht mehr gut gelernt. Auch für viel mehr Epochen nicht.
- Ansonsten:
- Man bekommt ca. 8 Kurven die sinnvoll aussehen und als trainingsdaten genutzt werden könnten für 3PINN



# 24.07. Mittwoch 14:00 Meeting
Alle Fragen die ich im Moment habe:
## Samples für die Isothermen
- (Wie) kann ich argumentieren, dass die Samples von den R-Kurven sinnvoll sind? (TODO: Was genau meine ich mit sinnvoll?)
- Was bedeutet es, dass die R-Kurven von verschiedenen Seeds qualitativ nicht von verschiedenen Zeitintervallen zu unterscheiden sind?
    - Müsste ich die verschiedenen Zeitintervalle noch mal mit festem Seed trainieren, um zu schauen, ob sich dann überhaupt was ändert?
- Wäre es besser uniform zwischen der niedrigsten Kurve und der größten Kurve zu sampeln? (Weil sonst kann es vor allem bei nicht so vielen Kurven zu recht großen Lücken kommen (TODO: Hier wäre ein Bild nicht schlecht))
- Wie bekomme ich es hin, dass 3PINN nahe Null mehr nach der FINN-Isotherme aussieht? (Also steil ansteigt)
    - Weil sonst tun auch die Quantile nicht wirklich gut, weil die Samples nahe Null dann quasi alle Outlier sind (TODO: Bild hierzu)
- 

-----


Ideen:
# Datenunsicherheit (Wie ändert sich Isotherme weil C Noise hat)
- FINN auf c -> 3pinn auf c -> Dann verschiedene c-Kurven (mit verschiedenen Quantilen (TODO: Gibt es genug?)) im c-3PINN-Interval mit FINN und festem Seeds lernen -> Ergibt viele Isotherme dessen Umhüllende meine Unsicherheit ist. (Wenn wir keinen Knick wollen 3pinn auf Samples der Kurven machen)
(Habe ich quasi schon nur jetzt mal mir festem Seed)
# Modellunsicherheit
A) 1. Zeitintervalle mit festem Seed = Modellunsicherheit aufgrund unvollständiger Daten
   2. Unterschiedliche Seeds = Modellunsicherheit aufgrund von Parameterunsicherheit
   3. C random (bei loss)


----
TODO:
- [x] 3pinn mit Zeitintervallen nimmt keinen Median im Moment
- [] Learining rate decay
- [] Loss: MAE vs MSE in 3pinn
- [] Zeit messen und vergleichen
- [x] C für den Loss random sampeln
- [] What if I use mean for the loss in FINN?
- [] Was wenn ich statt nen mean zu learnen bei 3pinn den FINN mean einfach nehme und mit ihm aufsplitte?
    - Das würde aber nicht dem Datensatz den ich gesampelt habe entsprechen. Die FINN Kurve kann ganz schön anders als der Datensatz aussehen.

-----
# Zeitmessungen:
FINN, 251 timesteps, 100 epochen, 26 spatial steps, 7.75 minutes
FINN, 251 timesteps, 100 epochen, 26 spatial steps, randomly masked c field for loss, 12 minutes