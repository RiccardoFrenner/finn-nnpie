# TODOs

- [] What if I use mean for the loss in FINN?
- [] Was wenn ich statt nen mean zu learnen bei 3pinn den FINN mean einfach nehme und mit ihm aufsplitte?
    - Das würde aber nicht dem Datensatz den ich gesampelt habe entsprechen. Die FINN Kurve kann ganz schön anders als der Datensatz aussehen.

-----

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
- [x] C für den Loss random sampeln
- [x] Loss: MAE vs MSE in 3pinn
- [x] Learining rate decay
- [] Zeit messen und vergleichen
- [] Von Kurven samplen und nicht uniform zwischen envelopes

- [] c_train anschauen ob bestimmte konzentrationen da wo envelope schmall ist häufiger oder weniger häufig vorkommt.


-----
# Zeitmessungen:
FINN, 30 timesteps, 100 epochs, 26 spatial steps, 5 minutes
FINN, 51 timesteps, 100 epochs, 26 spatial steps, 7.75 minutes
FINN, 51 timesteps, 100 epochs, 26 spatial steps, randomly masked c field for loss, 12 minutes

-----
11:30
- Vorstellung
- FINN nur ganz kurz mit Problemstellung
- 3pinn erklären
- Ansätze für Unsicherheiten vorstellen
- Ergebnisse

# Interesting Observation for Running Intervals
- It seems like only the first interval even converges (MSE < 1e-2)

TODO
- [] Compare Rs for running intervals epochs=100 and epochs=1000

----
# 20.08.24
Was ich vom Milestone mitgenommen habe:
- Wir shiften den Fokus von "FINN mit 3PINN kombinieren" zu "Wie kann man/Welche Arten gibt es, FINN zu stören und wie sehen die Isotherme dafür aus? Und danach 3PINN als post-processing anwenden."
- Welche Störungsart hat den größten Einfluss?
- Wie kann man noch lokale Minima sampeln? Reicht verschiedene Seeds oder sollte man noch am Löser etwas ändern? Möglich wären Art des Optimierers, Early Stopping,
- Langmuir statt Freundlich verwenden, da keine Singularität bei Null (und im Allgemeinen weil Langmuir physikalisch fundiert ist und Freundlich nicht (Freundlich hat auch unangenehme physikalische Einheiten))
- "Analyseunsicherheit" ist besseres Wort als "Modellunsicherheit", weil Modellunsicherheit sich auf das mathematische Modell, also die PDE bezieht.
    - Was ist aber wenn ich experimentelle Daten nehme? Habe ich dann nicht das Modell als dritte Unsicherheit immer dabei?
    - Entfernt verwandt: Ist R(c) eigenlich mathematisch eindeutig mit gegebener PDE und Randbedingungen?
- Datenunsicherheit entweder direkt mit experimentellen Daten oder mit Noise auf synthetischen. Parameter des Noises entweder durch Residuen mit exp. Daten oder von Laboranmerkungen (sollte es öffentlich auf Github? wohl geben) ableiten.
- Auch verschiedene Störungen zusammen sampeln (z.B. Noise auf Daten + Seeds (Braucht auch nicht mehr Samples, weil MC dimensionsunabhängig ist.)). Das kann dann mit MCMC von Timothy verglichen werden, da dies auch beide Unsicherheiten zusammen betrachtet.
- Die zwei Schnittpunkte der Isothermen in allen Methoden könnten Ursache von globaler, hidden Features dieses Problems (oder dieser Daten? (Man könnte mal schauen ob für andere Daten die gleichen Punkte auftreten.)) sein.
- Warum sind alle Rs bei großem c größer als Freundlich R? Dazu wurde was gesagt, aber weiß nicht sicher mehr was. (Ich erinnere mich vage, dass möglicherweise die potenziell unterschiedlichen Vorwärtslöser, die in FINN und für die Trainingsdaten verwendet wurden, als Ursache dafür postuliert wurden.)

-----
# TODO
- [] Alle FINNs neu machen mit selfgen c
    - [] Freundlich
    - [x] Langmuir
- [x] FINN mit Langmuir scheint überhaupt nicht zu funktionieren. Siehe `analyze_finn_forward_code_differences.ipynb`. Weder mit github c_langmuir noch mit FINN forward c_langmuir.
    - **Das Problem war mein Forward FINN Code und dass das github langmuir c scheinbar andere Parameter für die langmuir Isotherme verwendet hat. Ersichtlich aus dem riesigen Unterschied zwischen meiner Langmuir und den github c predicted langmuirs trotz kleinem c_pred error.**
    - [o] Schauen ob es mit dem originalen Code funktioniert.
        - [] Cognitive Modelling: Hier wird die Datengenerierung explizit implementiert.
            - [x] Wie ist die Datengenerierung implementiert?
                - **solve_ivp** von scipy. Wird aber nur für die Datengenerierung verwendet, nicht für FINN. Also auch unterschiedliche forward-codes.
            - [] Nur zur Sicherheit schaue ich mir hier auch noch mal die Isotherme an
                - [] Langmuir scipy
                - [] Freundlich scipy
                - [] Langmuir FINN forward (meine Modifikation)
                - [] Freundlich FINN forward (meine Modifikation)
        - [] Timothy: Hier könnte ich den Code wieder um die Datengenerierung erweitern wie schon mal gemacht.
            - [] Ergebnisse des umgeschriebenen Codes
            - [] Ergebnisse für Langmuir von github
- [x] Neues c_train erzeugen mit FINN forward Löser (statt dem bisherigen von Github c_train.csv)
    - **Alles egal, hatte falschen FINN forward Code. Siehe oben.**
    - **Tja, der FINN forward Code erzeugt ganz schön andere Ergebnisse. Linf-Fehler liegt für Langmuir bei ca. 0,4. Für Freundlich konvergiert es gar nicht. Der Linf-Fehler zwischen FINN-langmuir und FINN-linear ist viel geringer (0,07).**
    - [] Wie sieht es mit dem Datengenerierendem Code aus dem Fehlerhaften Github aus?
    - [] Wieso konvergiert hier Freundlich nicht?
    - [] Was ist der Unterschied zwischen der Trainingsergebnissen der beiden?
    - [] Mit selfgen Langmuir braucht das Training Ewigkeiten. (100s für die meisten ersten Epochen statt den üblichen 20s)
- [] FINN Training für C mit Noise analysieren
    - [x] Habe ich mit gleichen Anfangsgewichten trainiert?
        - Ja, habe ich.
    - [x] Warum konvergieren so viele nicht?
        - [x] Wie definiere ich hier Konvergenz überhaupt? Die Fehler sind immer im Rahmen des Noises. MSE sagt mir ja nicht, ob ich immer noch einen guten Mean habe. Wenn die Daten unsicherer sind, ist auch der MSE größer, egal ob ich noch die gleich gute Mean Kurve bestimme. Deshalb würde es eigentlich mehr Sinn ergeben den MSE mit den ursprünglichen nicht-noisy Daten zu berechnen.
            - **Bei vielen Daten steigt MSE(Model, Reality) nicht wirklich an, weil das Model eben durch den gleichmäßigen Noise die Realität noch erkennen kann.**
            - **"MSE mit den ursprünglichen nicht-noisy Daten zu berechnen" ist MSE(Model, Reality). Bei FINN steigt dieser an, was aber an der geringen Datenmenge liegt. Was kann ich dann über die Konvergenz aussagen? Eigentlich bleibt mir nichts anderes übrig als trotzdem diesen MSE zu nehmen.**
        - **Es konvergieren so wenig, weil das Sigma für die meisten zu groß ist und wenig Daten vorhanden sind. Ich weiß, keine wirklich zufriedenstellende Antwort, weil FINN ja eigentlich ohne viele Daten auskommen sollte, aber was besseres ist schwer zu sagen, ohne die generelle Konvergenz von FINN unter verschiedenen Bedinungen zu analysieren.**
    - [x] Welche konvergieren überhaupt?
        - **Diese Frage hätte sich mit der obigen "Definition" von Konvergenz geklärt. Interessant ist nur noch, dass auch stark (von der analytischen) abweichende Isotherme zu Konvergenz führen.**
    - [x] Wieso kommen so krass unterschiedliche Isotherme raus? Wieso wird nicht immer die gleiche Isotherme gelernt, die die mittlere Konzentration lernt? Wenn ich nen Sinus mit unterschiedlichem Noise lerne, ist die Kurve ja auch immer gleich.
        - **Nur wenn genug Daten vorhanden sind und nicht overfitted wird. Sollte eigentlich für FINN beides kein Problem sein. (TODO: Warum also doch?)**
    - [] Wieso gibt es die eine R(c) Kurve die fast Steigung = 0 hat für c > 0.5?
        - **Diese Kurve gibt es nicht nur einmal, sonder häufig. Ich weiß nicht, ob sie immer identisch ist (TODO), aber MSE(pred, real) ist schon mal unterschiedlich zwischen ihnen.**
    - [] Welches Sigma wäre jetzt überhaupt realistisch?


# Meeting 29.08.24
## Milestone
- Fokuswechsel: Wie kann man FINN stören?
    - Welche Störungsart hat größten Einfluss?
    - Wie kann man lokale Minima noch sampeln?
        - Anfangsgewichte
        - Dropout?
        - Wahl des Optimierers / seiner Parameter?
- "Analyseunsicherheit" statt "Modellunsicherheit" als Wort verwenden.
    - Was wenn wir experimentelle Daten nehmen? Haben wir dann nicht auch Modellunsicherheit? (Vergleiche mit Ergebnissen durch Wechseln zu FINN forward solver)
- Entfernt: Ist R(c) eindeutig?

## Neue Ergebnisse
- Datenunsicherheit via Noise auf Synthetic C
- FINN als Forwärtslöser statt Daten von Github (+ Langmuir statt Freundlich)
- MSE(c_pred,c_train) für running intervals

## Weiteres Vorgehen
- Masterarbeit wolfgang 2000? Waterloo, sonst timothy (für noise parameter)
- Fehlerberechnung bei running intervals inkorrekt (nimmt nur subset).
- Matze Gültig (FINN mit PEFAS), Juni Schnitzler(Bayes NN mit FINN); 2019 Jahrgang haben Bachelorarbeiten drüber gemacht (FINN)
- Gliederung schreiben