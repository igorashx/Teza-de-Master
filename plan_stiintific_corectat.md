# Plan științific revizuit

## Titlu propus

**Modelarea dinamicii latente a activității neuronale: comparație între LFADS discret, Latent Neural ODE și Latent CfC, evaluare NLB, teste de robustețe la corupții artificiale și analiză geometrică.**

Modelul discret de referință va fi **LFADS canonic**, nu un VAE GRU mai simplu. Se folosește un generator GRU cu condiție inițială per trial, factori latenți și controller recurent care inferă intrări dependente de timp, cu distribuție observațională Poisson [Sussillo et al., 2016](https://arxiv.org/abs/1608.06315), [Pandarinath et al., 2018](https://doi.org/10.1038/s41592-018-0109-9). Comparația cu NODE și CfC este una între arhitecturi complete; nu identifică efectul cauzal al timpului continuu separat de diferențele de encoder, posterior și parametrizare.

## 1. Scop și întrebări de cercetare

Scopul principal este compararea empirică a trei **modele generative ale spike-counturilor binned**: LFADS, Latent Neural ODE și Latent CfC. Toate folosesc aceleași observații, splituri, distribuție de emisie Poisson și metrici; posteriorul, priorurile și obiectivul de antrenare rămân cele definite de fiecare arhitectură și sunt raportate explicit, nu forțate să fie identice. LFADS este modelul discret recurent de referință, iar NODE și CfC sunt modelele cu dinamică dependentă de timp continuu. Diferențele observate susțin concluzii despre aceste implementări complete în protocolul stabilit, nu superioritate universală și nici efect izolat al discretizării. Nu se va pretinde că vreun model recuperează unic „adevărata” stare neuronală: reprezentările latente sunt neidentificabile până la transformări ale coordonatelor. Geometria latentă va fi descrisă ca proprietate a modelului învățat.

### Întrebări primare, pre-înregistrate

1. În condiții de selecție și evaluare identice și la bugete de capacitate raportate, cum diferă LFADS discret, Latent Neural ODE și Latent CfC în co-smoothing NLB?
2. Pentru `MC_Maze`, diferă performanța oficială de decodare a vitezei mâinii obținută din ratele inferate?
3. Pe un set Lorenz cu counturi parțial mascate, cum diferă reconstrucția counturilor neobservate între LFADS discret și modelele care primesc explicit intervalele temporale?

### Întrebări secundare, exploratorii

1. Cum diferă dimensiunea efectivă, continuitatea temporală și separabilitatea condițiilor în reprezentările latente?
2. Ce proprietăți locale ale hărților dinamice învățate sunt stabile între seed-uri?
3. Care este compromisul între calitate, număr de parametri, număr de evaluări ale dinamicii și timp de execuție?
4. Cum se degradează performanța predictivă și cât de mult se schimbă ratele inferate la thinning și contaminare Poisson a spike-counturilor observate?
5. Care este compromisul măsurat între calitate predictivă, timpul de antrenare, latența de inferență și memoria maximă?

Nu se va formula în avans concluzia că modelele continue sunt „mai bogate”, „mai stabile” sau „mai analizabile”. Acestea sunt ipoteze exploratorii ce pot fi susținute sau respinse de rezultate.

## 2. Corecții științifice obligatorii

| Formulare eliminată | Formulare corectă și operațională | Fundament |
|---|---|---|
| „BiGRU ignoră timpul neregulat.” | Pentru eșantioane binned regulate, BiGRU modelează ordinea binurilor, dar nu primește explicit intervalul dintre observații și nu poate recupera timpul pierdut în interiorul binului. | Un ODE-RNN este motivat pentru goluri **între observații** cu timpi arbitrari, nu pentru simple spike-counturi la pași egali [Rubanova et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html). |
| „B2/B3 au encoderul teoretic optim.” | ODE-RNN și CfC sunt encodere candidate. Superioritatea lor este o ipoteză testată, nu o proprietate demonstrată. | ODE-RNN a fost evaluat pentru serii neregulat eșantionate; NLB furnizează pentru benchmark predicții în binuri de 5 ms [Rubanova et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html), [Pei et al., 2021](https://doi.org/10.48550/arXiv.2109.04463). |
| „CfC are același câmp vectorial $f(z)$ ca NODE.” | Un CfC implementat ca celulă `ncps` este accesat practic ca o hartă $F_{\Delta t}$ dependentă de pasul temporal. Analiza sa principală este discretă, prin Jacobianul hărții. Nu se vor raporta divergență, nulcline sau exponenți Floquet ai unui $f$ inexistent în implementare. | CfC este o aproximație în formă închisă a dinamicii LTC și evită solverul numeric, nu este API-ul unui vector field Neural ODE [Hasani et al., 2022](https://doi.org/10.1038/s42256-022-00556-7); [documentația `ncps`](https://github.com/mlech26l/ncps). |
| „LDA $R^2$.” | Pentru condiții discrete se raportează balanced accuracy macro și matricea de confuzie a unui LDA, pe trialuri de test. $R^2$ este rezervat țintelor continue. | Definiția statistică a lui $R^2$ nu este o metrică de clasificare. |
| „DMFC_RSG: rt-$R^2$ oficial.” | Metrica secundară NLB pentru `DMFC_RSG` este corelația Pearson dintre viteza neuronală medie pe trial și intervalul produs $t_p$. O regresie Ridge spre reaction time poate fi raportată doar ca analiză suplimentară, etichetată neoficială. | [Pei et al., 2021, secțiunea 3.3](https://arxiv.org/html/2109.04463v4). |
| „Un test t pereche cu 3 seed-uri demonstrează $p<0.05$.” | Se folosesc minimum 10 seed-uri independente pentru analiza principală; se raportează diferența medie, intervalul de încredere bootstrap 95%, toate valorile pe seed și $p$ ajustat Benjamini-Hochberg. | Cu trei perechi, testul de permutare exact bilateral are doar opt semne posibile și nu poate produce $p<0.05$; controlul FDR este definit de [Benjamini și Hochberg, 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). |
| „Dacă $\lambda_1<0$, Lorenz este recuperat incorect.” | Pentru Lorenz se verifică mai întâi estimatorul pe ecuațiile Lorenz cunoscute. Pentru modelul învățat se raportează $\lambda_1$ cu interval de încredere și se evaluează dacă păstrează semnul pozitiv; concordanța numerică cu valoarea de referință este secundară și are toleranță declarată. | Parametrii canonici sunt din [Lorenz, 1963](https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2). Exponenții estimați depind de timp, integrare și estimare finită. |

## 3. Modele comparate

Toate modelele primesc spike-counturi $x_{1:T}\in\mathbb{N}^{N}$ la binuri de $\Delta t=5$ ms, produc rate $r_{1:T}>0$ și folosesc aceeași distribuție observațională. $h_t$ denumește starea/factorii specifici fiecărei arhitecturi: $h_t=z_t$ pentru M1/M2 și factorii readout-ului pentru LFADS.

$$
x_{t,n}\sim \operatorname{Poisson}(r_{t,n}\Delta t),\qquad
r_t=\exp(W_{\mathrm{out}}h_t+b_{\mathrm{out}}).
$$

În cod se folosește stabilizare numerică a log-ratei înainte de exponențială, iar ratele sunt raportate în Hz. Distribuția observațională, datele, spliturile și metricile sunt comune; readout-ul, posteriorul, priorurile, regularizarea KL și obiectivul complet sunt cele ale fiecărei arhitecturi și se documentează separat.

### M1 — Latent Neural ODE

- Posterior: encoder BiGRU care produce $\mu$ și $\log\sigma^2$ pentru $q(z_0\mid x_{1:T})$; starea inițială este eșantionată prin reparametrizare.
- Dinamică autonomă: $\dot z(t)=f_\theta(z(t))$, cu $f_\theta$ MLP Tanh. Timpul $t$ nu este intrare în modelul primar, pentru ca punctele fixe să fie bine definite.
- Integrare la instanțele de bin: solver adaptiv explicit, toleranțe predefinite și logarea numărului de function evaluations (NFE).

Un model condiționat de timp sau de covariate se va trata separat. Pentru $\dot z=f(z,t)$, un punct fix al lui $f(z)$ nu are în general interpretare autonomă; analiza corectă necesită un sistem augmentat sau un timp înghețat, declarat explicit.

### M2 — Latent CfC

- Posterior: același encoder BiGRU și aceeași distribuție $q(z_0\mid x_{1:T})$ ca la M1, pentru a menține constantă inferența de bază în comparația NODE--CfC.
- Dinamică: $z_{t+1}=F_{\mathrm{CfC},\Delta t}(z_t)$, cu intrare externă nulă după inițializare și `timespans=\Delta t` transmisă explicit conform API-ului versiunii fixate a `ncps`.
- Citire Poisson: comună.

CfC este tratat ca model cu pas temporal variabil, nu ca echivalent direct al unui Neural ODE. Reproducerea cere versiunea exactă a `ncps` și un test unitar care verifică faptul că schimbarea `timespans` schimbă ieșirea celulei.

### Paritate de capacitate

Paritatea de parametri este o operaționalizare limitată a capacității, nu garanția unei capacități funcționale egale. Înainte de selecția hiperparametrilor se fixează dimensiunea latentă principală $d=16$ și se enumeră într-o grilă YAML configurațiile candidate ale fiecărui model. Se calculează exact numărul de parametri antrenabili separat pentru encoder/controller, dinamică/generator, decoder/readout și total. Se caută configurații M1--M3 cu număr total în ±10% față de o țintă comună realizabilă, minimizând abaterea maximă; dimensiunile generatorului, controllerului, factorilor și intrărilor inferate LFADS se includ în grilă. Parametrii și config-urile selectate se publică. Dacă potrivirea nu este realizabilă, rezultatele se raportează cu numărul de parametri și nu se formulează concluzia „la capacitate egală”; comparația de referință rămâne validă ca evaluare de modele complete. O țintă de 50.000 ± 5.000 parametri nu se impune fără verificare pe toate arhitecturile. Același număr de parametri nu echivalează capacitatea funcțională sau ușurința de optimizare.

Grila, regula de selecție și numerele exacte de parametri se îngheață înainte de accesul la validation și test. Se raportează și o analiză de sensibilitate cu arhitecturi de dimensiuni apropiate, deoarece același număr de parametri nu egalizează biasul inductiv sau ușurința optimizării.

### M3 — LFADS canonic, model discret de referință

LFADS este implementat conform arhitecturii publicate: encoder bidirecțional pentru posteriorul condiției inițiale a generatorului, encoder al secvenței urmat de controller recurent care inferă intrările, generator GRU care primește condiția inițială și intrările inferate, factori latenți și readout Poisson. Pentru NLB regulat se folosește configurația `lfads-torch` fără mască temporală. Aceasta nu este automat aceeași configurație cu baseline-ul **AutoLFADS** din NLB; rezultatul se numește LFADS (`lfads-torch`), iar revendicarea unei reproduceri AutoLFADS cere aceeași variantă de arhitectură, căutare PBT, preprocesare și splituri. În experimentul Lorenz cu date lipsă se folosește varianta explicit etichetată **LFADS-mask**: counturile lipsă nu sunt confundate cu zero observat, encoderul primește masca binară, iar likelihood-ul include doar observațiile prezente. Configurația LFADS NLB și adaptarea LFADS-mask se disting în YAML și în rezultate. Grila LFADS include coordinated dropout activat/dezactivat ca regularizare predeclarată, selectată numai pe validation; se raportează ablația pereche, deoarece autoencoderele spike pot învăța o cale de tip identity care reconstruiește counturile de intrare fără a recupera ratele subiacente [Keshtkaran și Pandarinath, 2019]. Această regularizare de train nu se confundă cu stress-test-ul de perturbare al intrărilor. Nu se va numi „LFADS” o simplă pereche BiGRU+GRU fără controller și posteriorul intrărilor.

Intrările inferate sunt variabile latente explicative ale modelului, nu măsurători ale inputurilor biologice; separarea dintre dinamica generatorului și intrările inferate este neidentificabilă în general și poate atribui intrărilor variații pe care generatorul nu le-a modelat [Sussillo et al., 2016]. Nici robustețea la contaminare artificială nu este interpretată ca identificare a inputurilor reale.

LFADS este membru principal al comparației, nu un comparator secundar. Estimandul este diferența dintre arhitecturi complete sub aceleași date, emisie Poisson, splituri, selecție validation și bugete de capacitate/compute raportate. Rezultatul nu izolează efectul controllerului și nu demonstrează că discreția în sine cauzează o diferență; acest lucru se va declara în întrebări, rezultate și concluzii.

## 4. Ablația encoderelor NODE/CfC

Ablația are două regimuri, fiecare cu dinamică, decoder și buget de parametri fixate.

| Regim | M1: Neural ODE | M2: CfC | Interpretare admisă |
|---|---|---|---|
| C: encoder comun | BiGRU | BiGRU | Efectul dinamicii NODE versus CfC condiționat de encoderul BiGRU și celelalte setări fixate. |
| E: encoder alternativ | ODE-RNN | CfC bidirecțional | Efect empiric al înlocuirii encoderului în cadrul fiecărei familii; nu „optimalitate” și nu cauzalitate biologică. |

Pentru M1 și M2 se face contrastul pereche C versus E cu **aceleași seed-uri**, aceleași trialuri și măști, aceeași ordine de mini-batch-uri și aceeași dinamică generativă în interiorul fiecărui model. Selecția encoderelor se face pe validation cu un buget identic, apoi configurațiile se îngheață înainte de test. Astfel, schimbarea intenționată este encoderul; se raportează diferența de parametri și costul de calcul. Această ablație nu include LFADS și nu izolează efectul discret-versus-continuu între familii.

Pe NLB, unde binurile sunt regulate, această ablație nu testează un avantaj al timestampurilor neregulate. Ea testează doar utilitatea inductive bias-ului ODE-RNN/CfC față de encoderul BiGRU în modelele NODE/CfC. Performanța comparativă a LFADS față de modelele continue rămâne un contrast între arhitecturi complete. Efectul explicit al gap-urilor se testează în experimentul sintetic din secțiunea 5.

## 5. Date și protocol de separare

### 5.1 Lorenz sintetic

Se folosesc ecuațiile Lorenz autonome cu $\sigma=10$, $\rho=28$, $\beta=8/3$. Pentru fiecare trial:

1. se alege o condiție inițială dintr-o distribuție declarată;
2. se execută un burn-in declarat înainte de fereastra observată;
3. se integrează cu un solver de referință și toleranțe stricte;
4. se generează 50 rate prin $r(t)=\exp(Cz(t)+d)$; $C$ și $d$ sunt fixate o dată pe dataset;
5. se calibrează $d$ pe setul de antrenare astfel încât media populației să fie 5--20 Hz, apoi se eșantionează spike-counturi Poisson de 5 ms.

Trialurile, nu timepointurile, se împart înainte de antrenare în train/validation/test, de exemplu 70/15/15%, cu seed de date distinct de seed-urile de antrenare. Nu este permisă împărțirea aleatoare a binurilor din același trial.

Sunt două variante declarate dinainte:

- **L-regulat**: toate observațiile la 5 ms. Aici C versus E este o ablație arhitecturală, fără afirmație despre irregularitate.
- **L-neregulat**: se generează counturi Poisson la fiecare bin de bază de 5 ms, apoi se maschează binuri temporale întregi printr-un mecanism MCAR Bernoulli cu probabilitatea de păstrare fixată înainte; intervalele dintre binurile păstrate au distribuție geometrică în unități de 5 ms. Nu se agregă counturile. Pentru M1/M2, encoderul primește counturile observate și timestampurile lor, iar ODE-RNN/CfC propagă starea folosind gap-ul real; generatorul produce rate la toate binurile de 5 ms. Pentru M3, intrarea encoderului păstrează grila regulată completă, counturile absente sunt înlocuite cu zero numai în tensorul de intrare și o mască binară explicită distinge lipsa observației de un count observat egal cu zero; generatorul discret evoluează la fiecare bin. În toate modelele, likelihood-ul de antrenare include numai binurile observate. Counturile mascate nu sunt folosite la inferență sau antrenare și se accesează numai la evaluare, unde held-out Poisson log-likelihood/bps se calculează pe binurile mascate. Pentru fiecare set se generează mai multe realizări de mască; aceleași măști și aceleași counturi Poisson se folosesc pentru toate arhitecturile și seed-urile. Măștile de train se pot re-eșantiona determinist pe epocă, iar măștile validation/test se fixează și se salvează înaintea selecției checkpointului. Acesta testează reconstrucția la date lipsă și compară modele complete; nu este doar o intervenție asupra encoderului și nu izolează cauzal timpul continuu.

Validarea generatoarelor Lorenz este în două etape. Mai întâi algoritmul QR este rulat asupra sistemului Lorenz analitic până la convergență; rezultatul, pasul de integrare, timpul de burn-in și timpul total sunt salvate. Apoi, pentru dinamica Neural ODE se estimează spectrul pe traiectorii lungi ale modelului. Criteriul minim este un interval de încredere al lui $\lambda_1$ complet pozitiv, nu egalitatea exactă cu o singură cifră publicată. O eroare relativă față de estimatorul de referință este raportată, dar nu este optimizată în timpul antrenării.

Pentru compararea geometriei cu adevărul Lorenz, latentele sunt aliniate post-hoc cu o mapare afină/Procrustes estimată **numai pe train**, apoi evaluată pe test. Se raportează eroarea de traiectorie aliniată și corelația distanțelor pereche ca măsuri descriptive, cu sensibilitatea la metoda de aliniere. Un interval complet pozitiv pentru $\lambda_1$ al generatorului învățat nu demonstrează recuperarea dinamicii Lorenz: un model cu latent mai mare poate avea dinamici suplimentare, iar alinierea observată nu demonstrează conjugare dinamică. Verificarea operațională combină likelihood predictiv Poisson pe observații test, predicție liberă în spațiul observațiilor la mai multe orizonturi, eroarea geometriei aliniate și diagnosticul Lyapunov. Aceste măsuri se raportează separat; nu se comprimă într-un verdict binar de „recuperare Lorenz”. Estimatorul Lyapunov se validează întâi pe sistemul Lorenz analitic.

### 5.2 NLB `MC_Maze`

- Se utilizează API-ul și spliturile publicate în versiunea fixată a `nlb_tools`; nu se reconstruiesc splituri proprii.
- `MC_Maze-S` are 100 trialuri train, `MC_Maze-M` 250 trialuri train, iar fiecare are 100 trialuri test; ele provin din sesiuni diferite, nu sunt subseturi imbricate. Nu se transferă hiperparametri aleși pe M către rezultatul S sau invers fără a declara transferul [Pei et al., 2021, Tabelul 4](https://arxiv.org/html/2109.04463v4).
- Metrica primară este co-smoothing bps pe neuroni held-out. Pentru inferența de test se folosesc numai neuronii held-in disponibili.
- Metricile NLB secundare sunt: $R^2$ pentru viteza mâinii prin decoder Ridge conform protocolului NLB, PSTH-$R^2$ și forward-prediction bps acolo unde este aplicabil. Forward prediction este raportată ca metrică secundară, fiind justificată pentru activitate previzibilă, nu ca probă că sistemul biologic este autonom.

Structura rotațională a activității populației în sarcini de reaching este documentată, inclusiv prin analiza jPCA [Churchland et al., 2012]. Planul testează dacă ratele inferate/generate de fiecare model păstrează această structură și cum variază între arhitecturi. Dovezile din activitatea populației nu implică însă, prin ele însele, existența unui ciclu limită autonom în generatorul latent: traiectoriile pot fi tranzitorii, condiționate de sarcină sau dependente de intrări. Prin urmare, rotația este o analiză exploratorie susținută de literatura anterioară, iar ciclul limită rămâne o afirmație distinctă care necesită verificarea dinamicii libere și a închiderii orbitale.

### 5.3 NLB `DMFC_RSG`

- Se utilizează spliturile NLB: 1006 trialuri train, 283 test, 40 neuroni held-in și 14 held-out [Pei et al., 2021, Tabelul 7](https://arxiv.org/html/2109.04463v4).
- Primar: co-smoothing bps. Secundar oficial: corelația Pearson dintre viteza neuronală medie pe trial și $t_p$.
- PSTH-$R^2$ se calculează cu protocolul NLB. O regresie directă din latent spre $t_p$ este o analiză suplimentară și nu este confundată cu metrica benchmarkului.

Nu se presupune „rampă spre puncte fixe”. DMFC_RSG implică atât structură internă, cât și influență de sarcină; o dinamică autonomă poate fi o aproximație utilă, nu un model demonstrat al circuitului [Pei et al., 2021](https://neurallatents.github.io/datasets).

### 5.4 Test de robustețe la zgomotul intrării

Testul de robustețe folosește modele antrenate pe counturi curate și perturbă numai spike-counturile held-in oferite la inferența de test; counturile held-out de referință și comportamentul rămân curate și neschimbate. Se evaluează separat două perturbări ale intrării, aplicate acelorași trialuri pentru toate modelele:

- **Thinning:** $x'_{t,n}\sim\operatorname{Binomial}(x_{t,n},q)$, cu $q\in\{1,0.75,0.50,0.25\}$. Aceasta modelează pierderea aleatorie a unor spike-uri observate, nu schimbarea intrinsecă a ratei neuronale.
- **Contaminare Poisson aditivă:** $x'_{t,n}=x_{t,n}+\eta_{t,n}$, cu $\eta_{t,n}\sim\operatorname{Poisson}(\alpha\bar r_n^{train}\Delta t)$ și $\alpha\in\{0,0.1,0.5\}$, unde $\bar r_n^{train}$ este rata medie estimată exclusiv din train. Se raportează separat și combinațiile thinning + contaminare, fără a le confunda cu thinning pur.

Se folosesc cel puțin 10 realizări independente de perturbare per trial și nivel; aceleași realizări ale counturilor corupte se reutilizează pentru toate modelele și seed-urile, pentru comparație pereche. Selecția checkpointului, a hiperparametrilor și a nivelurilor se face fără test: nivelurile de perturbare sunt fixate aici, iar validation servește doar verificării implementării. Nu se reantrenează pe date corupte în analiza primară.

Endpointul de performanță este co-bps pe counturile held-out curate (și velocity-$R^2$ NLB pentru `MC_Maze`), raportat față de condiția curată pentru fiecare model. Separat, stabilitatea numerică a reconstrucției se măsoară prin corelația și NRMSE dintre ratele inferate din intrarea curată și cea coruptă, pe aceiași neuroni held-out, trialuri și binuri. NRMSE se normalizează prin rata medie curată de train a neuronilor held-out; epsilon-ul numeric se fixează în configurație. Acordul dintre predicții nu este tratat drept acuratețe: poate fi mare și când ambele predicții sunt greșite. Se raportează simultan scorul față de counturile held-out curate și schimbarea față de inferența curată. Rezultatele sunt analiză secundară de robustețe la corupții artificiale, nu afirmații despre distribuția zgomotului biologic sau stabilitatea funcțională a circuitului.

## 6. Antrenare și selecție fără leakage

Modelele folosesc aceleași spike-counturi și aceeași emisie Poisson, dar ELBO-urile sunt arhitecturale, nu impuse identice. Pentru M1/M2, cu masca observațiilor $a_{t,n}\in\{0,1\}$:

$$
\mathcal{L}_{M1/M2}=\mathbb{E}_{q_\phi(z_0\mid x_{obs},a)}\left[\sum_{t,n}a_{t,n}\log\operatorname{Poisson}(x_{t,n};r_{t,n}\Delta t)\right]
-\beta_0 D_{KL}\!\left(q_\phi(z_0\mid x_{obs},a)\Vert p(z_0)\right).
$$

Pentru LFADS, notând cu $g_0$ condiția inițială și cu $u_{1:T}$ intrările inferate:

$$
\mathcal{L}_{LFADS}=\mathbb{E}_{q_\phi(g_0,u_{1:T}\mid x_{obs},a)}\left[\sum_{t,n}a_{t,n}\log\operatorname{Poisson}(x_{t,n};r_{t,n}\Delta t)\right]
-\beta_gD_{KL}(q(g_0\mid x_{obs},a)\Vert p(g_0))
-\beta_u\sum_tD_{KL}(q(u_t\mid u_{<t},g_0,x_{obs},a)\Vert p(u_t\mid u_{<t})).
$$

Pentru NLB, $a$ reflectă disponibilitatea counturilor conform protocolului held-in/held-out; pentru Lorenz mascat, likelihood-ul de train folosește numai binurile observate. Counturile mascate nu contribuie la loss și se accesează numai la evaluare.

- Selecția checkpointului și early stopping folosesc exclusiv validation co-bps.
- Testele NLB au devenit publice în ianuarie 2026, iar EvalAI nu mai acceptă submisii noi. Prin urmare nu se planifică o submisie oficială și nu se revendică un rang nou pe leaderboard. Se fixează modelul, codul, hiperparametrii și criteriile de evaluare înainte de a rula local datele publice de test; se consemnează că acestea nu mai sunt un test ascuns și riscul de supraajustare post-publicare. Se raportează scorurile locale cu versiunea fixată `nlb_tools` și, pentru context, scorurile publice anterioare numai dacă datasetul, splitul și metrica sunt identice. Se păstrează un split intern din train exclusiv pentru dezvoltare, fără a-l prezenta drept evaluare oficială NLB.
- Hiperparametrii sunt aleși pe Lorenz train/validation și pe validation NLB, nu pe test. `MC_Maze-S` este pilot, nu set de validare pentru `MC_Maze-M`, deoarece sunt sesiuni distincte.
- Fiecare model primește același buget de căutare a hiperparametrilor și aceeași regulă de early stopping pe validation, dar rata de învățare, annealing-ul și regularizarea pot fi specifice arhitecturii și se selectează numai pe validation. Dacă warm-up-ul unei configurații este 50 epoci, early stopping nu poate opri run-ul înainte de epoca 50; după aceea se aplică patience declarată. Configurațiile și bugetul se îngheață înainte de test.
- Se fixează și se salvează: seed Python/NumPy/PyTorch, versiuni pachete și CUDA, hardware, precizie, batch size, masca de neuroni, configurația YAML, hash-ul commitului, numărul de parametri, NFE și timpul wall-clock.

### Reguli de diagnostic și eșec

- Colapsul posteriorului se diagnostichează pe train/validation prin KL separat pentru stările inițiale și, la LFADS, intrările inferate, numărul dimensiunilor latente active după o regulă predeclarată, dispersia posteriorului și comparația cu priorul. Se păstrează curbele, nu doar un indicator binar. Pilotul poate selecta scheme arhitectură-specifice de annealing dintr-o grilă finită și cu buget egal; configurațiile finale se îngheață înaintea testului. Dacă un model rămâne colapsat, se raportează rezultatul, fără schimbarea retrospectivă a loss-ului doar pentru acel model.
- Pentru Neural ODE se fixează înainte de experimente un prag de solver `max_num_steps`/NFE în configurația versionată, ales pe pilotul train/validation și pe baza bugetului de resurse. Depășirea pragului, solverul care nu converge, NaN/Inf sau eșecul repetat de optimizare se înregistrează ca eșec de run. Run-ul nu se elimină silențios și nu se mărește pragul selectiv după inspectarea testului; se raportează frecvența eșecurilor, NFE și scorurile run-urilor valide, cu analiza de sensibilitate pentru prag dacă este necesară.
- O întrerupere de infrastructură se poate relua din checkpoint doar dacă starea optimizerului, RNG-urile și configurația sunt restaurate; altfel se reexecută același seed și se păstrează ambele loguri. O schimbare de cod/configurație după înghețarea protocolului se documentează ca deviație și impune rerularea întregii comparații pereche afectate.

Solverul Neural ODE este parte a modelului efectiv. Configurația primară poate fi `dopri5`, `rtol=1e-3`, `atol=1e-4`, însă `dopri5`, `rk4` și `euler` sunt comparate în ablația solverului cu aceleași setări în rest. Metoda adjoint și metoda directă nu se tratează ca identice: se loghează metoda de gradient, stabilitatea și memoria. Neural ODE permite explicit compromisul precizie-viteză prin solver [Chen et al., 2018](https://doi.org/10.48550/arXiv.1806.07366).

## 7. Metrici și reguli de calcul

### 7.1 Endpointuri primare

1. co-bps NLB pentru fiecare dataset;
2. $R^2$ velocity NLB pentru `MC_Maze`;
3. comparațiile pereche dintre M1--M3 în reconstrucția counturilor de 5 ms mascate în Lorenz, măsurate prin held-out Poisson log-likelihood/bps pe aceleași trialuri și măști;

### 7.2 Metrici secundare

- PSTH-$R^2$ și forward-prediction bps exact ca în `nlb_tools`.
- Participation ratio $\mathrm{PR}=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$, calculat din covarianța latentelor centrate. Se raportează atât PR, cât și $\mathrm{PR}/d$; comparația se face la același $d$.
- TwoNN intrinsic dimension, cu aceleași reguli de subsampling și whitening pentru toate modelele. Timepointurile autocorelate nu sunt tratate eronat ca eșantioane independente: se subsamplează uniform pe trialuri și se folosesc bootstrap-uri pe trialuri. Estimatorul TwoNN este definit de [Facco et al., 2017](https://doi.org/10.1038/s41598-017-11873-y).
- Smoothness normalizat: media lui $\lVert\tilde z_{t+1}-\tilde z_t\rVert_2/\Delta t$, după standardizarea $\tilde z$ pe train. Fără normalizare, o simplă rescalare a latentelor schimbă metrica.
- Consistență intra-condiție: distanța medie față de media condiției după aliniere temporală explicită; se raportează numai pentru condiții cu număr suficient de trialuri.
- Separabilitate: balanced accuracy a unui LDA antrenat pe train, cu regularizare aleasă pe validation și evaluat o dată pe test. Se raportează și baseline-ul de clasă majoritară.
- Stabilitatea predicției la corupție: corelația și NRMSE între ratele inferate din intrarea curată și cea perturbată, alături de co-bps și velocity-$R^2$ față de țintele curate. Acordul între două reconstrucții este raportat ca sensibilitate, nu ca metrică de acuratețe.
- Curbele de performanță la fiecare nivel de thinning/contaminare Poisson din secțiunea 5.4; ele se raportează față de intrarea curată și nu se folosesc la selecția checkpointului.
- Rotație pe `MC_Maze`: se aplică jPCA după procedura Churchland et al. (rate netezite, aliniere la evenimentul sarcinii, mediere pe trialuri în fiecare condiție de reaching și centrare conform procedurii publicate); orice abatere necesară pentru binurile/spliturile NLB se declară. Se raportează variația captată de planele jPCA, eroarea de predicție încrucișată a modelului rotațional față de modelul liniar fără constrângere skew-symmetric, frecvențele/constantele temporale și intervalele de incertitudine. Resamplingul se face pe trialuri înaintea medierii, cu seed-uri pereche; condițiile și fereastra temporală se fixează pe train/validation. Nu se impun praguri universale precum 30% variație explicată sau raport de putere >3; acestea nu sunt validate ca limite generale de prezență a rotației în literatura jPCA. Rezultatul testează prezența și stabilitatea structurii rotaționale, nu existența unui ciclu limită.

MARBLE este analiza geometrică secundară preplanificată, nu endpoint primar. Se folosește implementarea publică oficială și se fixează commitul, imaginea Linux, versiunile Python/PyTorch/PyG, preprocesarea, hiperparametrii și seed-urile. La preflight se verifică tipul efectiv de accelerator, memoria disponibilă, compilatorul și compatibilitatea CUDA/PyTorch Geometric; disponibilitatea resurselor nu se deduce doar din folosirea Codespaces. Dacă resursele permit, se execută pe GPU; altfel se folosește CPU numai dacă benchmarkul de runtime pe datele pilot satisface bugetul declarat. Înainte de experimente se rulează un exemplu oficial mic și un smoke test pe datele sintetice.

Pentru `MC_Maze`, fiecare punct este un vector de rate în spațiul comun al neuronilor; vectorul local este diferența de rate împărțită la intervalul temporal, cu tratarea explicită a gap-urilor. Se construiesc câmpuri separate pentru activitatea observată și pentru ratele estimate de fiecare model, grupate după condițiile experimentale conform ipotezei MARBLE de consistență dinamică. Modul embedding-aware este analiza primară, deoarece neuronii au aceeași ordine/coordonate de observație; modul embedding-agnostic este analiză de sensibilitate. Se raportează distanțele de transport optimal între distribuțiile MARBLE, interpretate ca similaritate a câmpurilor locale procesate de MARBLE, nu ca distanță absolută între „dinamicile cerebrale adevărate”. Se fixează dinainte cum sunt estimate ratele pentru neuroni held-out, cum se tratează ferestrele de mișcare și ce trialuri/condiții contribuie.

Se raportează conectivitatea grafului, dimensiunea locală, rata de subsampling, setările de difuzie și sensibilitatea la hiperparametrii principali. Modelul și toate transformările învățate se estimează pe train. Scorurile NLB confirmatorii nu folosesc MARBLE. Dacă versiunea oficială nu oferă transformare out-of-sample fără reantrenare, analiza pe test se etichetează explicit transductivă și nu este prezentată ca evaluare de generalizare; comparația principală MARBLE rămâne descriptivă, pe datele și regimul de potrivire definite înainte. Dacă instalarea sau criteriile minime de convergență eșuează, rezultatul se marchează neexecutat cu loguri și cauză. O analiză kernel locală de câmp vectorial poate fi raportată doar ca sensibilitate separată, nu ca substitut redenumit MARBLE.

### 7.3 Costul de calcul și latența

Toate măsurătorile comparative se fac pe același tip de Codespace/accelerator, cu aceeași versiune software, precizie și batch efectiv. Dacă ODE impune batch fizic mai mic, se folosește gradient accumulation pentru a păstra batch-ul efectiv și se raportează batch-ul fizic separat. Se măsoară distinct: preprocesarea, timpul optimizer per epocă, timpul total de antrenare până la checkpointul selectat, costul căutării hiperparametrilor în accelerator-hours, timpul de validare și memoria maximă CPU/GPU. Curbele validation co-bps în funcție de wall-clock și de bins procesate permit separarea vitezei de optimizare de performanța finală. Numărul configurațiilor candidate și plafonul de accelerator-hours pentru selecție sunt egale între familii și se fixează pe pilot înainte de test.

Inferența se măsoară separat pentru un singur trial și batch-uri comune de 16/64 trialuri, la lungimile reale ale dataseturilor, după 20 de warm-up-uri și minimum 100 de execuții temporizate. Dacă orice model nu poate rula unul dintre batch-urile respective pe acceleratorul fixat, se folosește cel mai mare batch comun tuturor modelelor și se raportează separat batch-ul maxim al fiecăruia. Pentru GPU se sincronizează acceleratorul înainte și după intervalul măsurat. Se raportează mediană, percentila 95, trialuri/secundă și memoria maximă, atât pentru un draw posterior cât și pentru media MC a unui număr comun $K^*$ de draw-uri. Pe validation, $K$ se dublează de la 8 până când eroarea Monte Carlo relativă a ratelor este sub 1% pentru toate modelele sau se atinge plafonul predeclarat $K=128$; se alege același $K^*$ pentru M1--M3 și se raportează eroarea Monte Carlo rămasă. $K^*$ se fixează înainte de evaluarea testului. Timpul model-only și timpul end-to-end (inclusiv pregătirea batchului, nu citirea fișierelor) se prezintă separat. Nu se transformă metricele de calitate și timp într-un scor compozit arbitrar.

## 8. Analiză dinamică compatibilă cu fiecare arhitectură


Toată analiza se aplică checkpointului selectat pe validation și se repetă pentru fiecare seed, nu doar pentru „cel mai bun” seed. Se lucrează în spațiul latent complet; graficele PCA/UMAP sunt numai vizualizări și nu sunt folosite pentru calculul Jacobianului.

| Analiză | LFADS (generator GRU discret) | Neural ODE autonom | CfC ca hartă $F_{\Delta t}$ |
|---|---:|---:|---:|
| Punct fix | $G(g^*,u=0)=g^*$, doar pentru generatorul liber | $f(z^*)=0$ | $F_{\Delta t}(z^*)=z^*$ |
| Stabilitate locală | la $u=0$, generatorul LFADS este stabil local dacă toți multiplicatorii lui $\partial G/\partial g$ satisfac $\lvert\mu_i\rvert<1$ | asimptotic stabil dacă toate valorile proprii ale $\partial f/\partial z$ au $\operatorname{Re}(\lambda_i)<0$ | stabil local dacă toți multiplicatorii lui $\partial F_{\Delta t}/\partial z$ satisfac $\lvert\mu_i\rvert<1$ |
| Exponenți Lyapunov | produs QR al Jacobienilor generatorului liber, `per step` | ecuație variațională/QR, pe unitatea de timp | produs QR al Jacobienilor hărții |
| Divergență, nulcline, Floquet continuu | nu | da, numai pentru $f(z)$ autonom | nu, decât dacă se implementează și validează separat un vector field echivalent |

Traiectoria LFADS reconstruită cu intrări inferate este un sistem discret forțat, în general neautonom. Analizele de punct fix și Lyapunov ale tabelului se aplică numai generatorului cu input fix $u=0$ (sau alt input constant declarat), nu trasei posteriori cu $u_t$ variabil. Comparația între acești generatori liberi este exploratorie și nu echivalează cu stabilitatea reconstrucției.

### Reguli pentru puncte fixe

- Se pornește optimizarea din stări latente observate și din puncte de referință aleatoare; numărul de porniri, toleranța reziduală și regula de deduplicare sunt declarate înainte.
- Pentru o hartă discretă se minimizează $\lVert F(z)-z\rVert^2$, nu $\lVert f(z)\rVert^2$.
- Un punct este raportat doar dacă reziduul normalizat este sub toleranța declarată și clasificarea sa este stabilă în cel puțin o proporție predefinită de seed-uri.

### Reguli pentru exponenții Lyapunov

Se raportează unitatea ($\mathrm{s}^{-1}$ pentru ODE, `per step` și conversie la $\mathrm{s}^{-1}$ declarată pentru hărți), durata totală, transientul eliminat, frecvența QR și intervalul bootstrap 95% pe traiectorii. Spectrele provenite din hărți și ODE nu sunt prezentate ca numeric identice; sunt comparate numai după conversia de unitate și cu această limitare în legendă.

Pentru Lorenz, se verifică estimatorul pe sistemul adevărat înainte de analiza modelelor. Pentru datele NLB, un exponent pozitiv sau negativ este o proprietate a generatorului ajustat, nu o demonstrație a haosului ori a disipării dinamiciilor cerebrale reale.

### Cicluri și bifurcații

Detectarea unui ciclu limită este distinctă de analiza jPCA și cere: orbită închisă în spațiul complet al stării generatorului liber, eroare de închidere normalizată sub toleranță fixată înainte, stabilitate orbitală verificată prin multiplicatori Floquet pentru ODE sau Jacobianul hărții, și replicare între seed-uri. Nu se cere ca traiectoriile experimentale de reaching să fie ciclice; durata lor finită și covariatele sarcinii pot produce rotații tranzitorii. O periodicitate în date, în jPCA sau în proiecția PCA nu este suficientă.

Nu se va numi „bifurcație” simpla scalare $f_\alpha=\alpha f$. Pentru un ODE autonom, această scalare pozitivă modifică în principal parametrizarea temporală și nu schimbă punctele fixe; prin urmare nu constituie un control fizic care poate identifica în general saddle-node, Hopf sau pitchfork. O analiză de bifurcație este permisă numai dacă se introduce un parametru de control cu sens experimental sau generativ explicit (de exemplu o covariată de condiție), se antrenează/validează un model condiționat și se analizează familia $f(z;c)$; altfel secțiunea este eliminată.

## 9. Plan statistic

- Se planifică minimum zece seed-uri de antrenare pentru endpointurile primare, dar zece nu garantează singur putere statistică suficientă. Numărul final se justifică înainte de test printr-o analiză de precizie/power bazată pe variația pilotului și pe o diferență minimă de interes declarată; nu se oprește sau extinde eșantionul după observarea semnificației pe test. Pilotul poate avea trei seed-uri și rămâne exclusiv pentru verificări tehnice/estimarea varianței, nu pentru concluzii confirmatorii.
- Pentru fiecare contrast pereche se raportează scorurile per celulă seed × trial, diferența medie și interval bootstrap 95%. Seed-urile de antrenare și trialurile de test sunt factori încrucișați, nu imbricate; se folosește un bootstrap de tip pigeonhole: la fiecare replică se resamplează independent cu înlocuire indicii seed-urilor și indicii trialurilor, apoi se calculează media diferențelor pereche pe produsul cartezian al indicilor extrași. Aceiași indici seed și trial sunt folosiți pentru ambele modele ale contrastului. Dacă scorul benchmarkului nu este definit pe trial, unitatea de resampling și formula de agregare se fixează pe baza metricii oficiale înainte de deblocarea testului; binurile temporale nu sunt tratate ca observații independente. Se publică algoritmul, numărul de replici, seed-ul bootstrap și nivelurile de resampling [Owen, 2007]. Testul de permutare pereche se aplică diferențelor agregate la nivelul seed-ului, nu binurilor.
- Familiile de teste primare sunt ajustate separat prin Benjamini-Hochberg la $q=0.05$. Metricile exploratorii sunt etichetate exploratorii, cu intervale de încredere și fără limbaj confirmator.
- Pentru robustețe, seed-urile modelului și trialurile se resamplează ca factori încrucișați, iar replicile de perturbare se resamplează în interiorul fiecărui trial; aceeași realizare coruptă se compară între modele. Intervalele pentru metricile de zgomot descriu incertitudinea între modele, trialuri și realizările perturbării, nu tratează binurile ca replici independente.
- Co-bps se compară în cadrul aceluiași dataset; valorile absolute nu se compară între dataseturi cu populații și rate diferite, conform avertismentului NLB.
- Efectul este interpretat prin mărime și interval, nu exclusiv prin pragul $p<0.05$. Pragul practic de interes pentru co-bps, dacă este folosit, se declară înainte și nu înlocuiește intervalul de încredere.

## 10. Ordinea experimentelor și criterii de trecere

1. **Teste unitare, LFADS și Lorenz de referință.** Se verifică dimensiuni, likelihood Poisson cu observații mascate, transmiterea $\Delta t$, implementarea LFADS completă pe un batch mic, estimatorul QR pe Lorenz analitic și smoke test-ul MARBLE. Se validează pe toy data mascarea encoderului și excluderea counturilor mascate din loss. Nu se trece mai departe dacă aceste referințe sau teste eșuează.
2. **Lorenz regulat și mascat.** Se confirmă distribuțiile counturilor, splitul pe trialuri, scorul Poisson pe counturile mascate, comportamentul la mai multe rate de mascare și generarea liberă. Estimatorul Lyapunov se interpretează numai pentru generatorul autonom NODE și se raportează ca diagnostic secundar, nu criteriu de trecere universal pentru LFADS/CfC.
3. **Pilot `MC_Maze-S`, trei seed-uri.** Se verifică compatibilitatea cu formatul NLB, masca held-in/held-out, lipsa NaN, KL posterior, NFE și costul memoriei. Nu se publică inferență statistică din pilot.
4. **Comparația confirmatorie pe NLB.** Se rulează M1--M3 pe `MC_Maze-M` și `DMFC_RSG`, la buget de selecție și de capacitate predeclarat; checkpointurile sunt selectate numai pe validation. Toate scorurile, timpul, memoria și abaterile sunt raportate pe fiecare seed.
5. **Ablații declarate.** Se fac după rezultatul principal: paritate de capacitate, solver NODE, dimensiune latentă și ablația encoderelor BiGRU versus ODE-RNN/CfC din secțiunea 4. Sweep-ul de dimensiune nu schimbă retrospectiv rezultatul principal la $d=16$.
6. **Robustețe și eficiență.** După înghețarea checkpointurilor și fără reselectare, se rulează perturbările pereche de thinning/contaminare și benchmarkurile de timp. Se raportează degradarea co-bps, stabilitatea ratelor inferate, curbele quality-versus-wall-clock și latența de inferență.
7. **Analize geometrice secundare.** Se rulează MARBLE în Codespace-ul Linux fixat, jPCA pe `MC_Maze`, geometria latentă și analiza locală a hărților. Aceste rezultate nu schimbă endpointurile confirmatorii și nu sunt folosite la alegerea checkpointurilor.

## 11. Output minim și raportare

1. Un tabel pentru fiecare dataset și regim, cu model M1--M3, seed, config hash, număr parametri, co-bps și metrici NLB aplicabile, timp/epocă, timp total de train, latența mediană/p95, throughput, memorie GPU, NFE unde este aplicabil (altfel `N/A`), statusul run-ului și motivul oricărui eșec.
2. Un tabel de contraste pereche LFADS--NODE, LFADS--CfC și NODE--CfC: diferență, interval 95%, $p$ brut, $p$ FDR, cost de calcul și diferență de parametri. Un tabel separat prezintă ablația C versus E pentru M1/M2.
3. Un tabel separat de analiză dinamică, cu tipul corect de sistem (`map`/`ODE`), unitatea exponentului, reziduurile punctelor fixe și intervalele de încredere. Celulele neaplicabile rămân `N/A`, nu `0`.
4. Curbe ELBO, co-bps și KL pentru fiecare seed; pentru LFADS se trasează separat KL-ul condiției inițiale și KL-ul intrărilor inferate. Graficele agregate arată media și intervalul, plus punctele individuale.
5. Figuri de traiectorii, PCA/jPCA/UMAP numai cu seed, split, normalizare și metodă de aliniere în legendă. UMAP nu este folosit ca probă cantitativă de separare.
6. Curbe comparative co-bps/velocity-$R^2$ în funcție de wall-clock și bugetul de căutare; grafice de latență și throughput la batch-urile declarate; curbe co-bps, NRMSE și corelație a ratelor inferate în funcție de thinning/contaminare, cu intervale de încredere și puncte per seed.
7. Un document `conclusions.md` care separă: rezultatele confirmatorii între arhitecturi complete, ablațiile de encoder, robustețea la perturbări artificiale, eficiența computațională, analizele secundare (MARBLE/jPCA), limitările și deviațiile. Se precizează explicit că testul NLB a fost public la momentul analizei și că nu a existat submisie nouă la leaderboard.

## 12. Afirmații finale permise și nepermise

Sunt permise formulări precum: „în acest protocol, comparațiile pereche dintre LFADS, NODE și CfC au dat diferențele și intervalele raportate”; „encoderul ODE-RNN a schimbat scorul de reconstrucție pe counturile Lorenz mascate în condițiile testate”; sau „thinning-ul la nivelul $q$ a redus scorul held-out cu diferența raportată”. Aceste rezultate nu izolează prin ele însele efectul cauzal al timpului continuu și nu generalizează la orice date neuronale.

Nu sunt permise fără dovezi suplimentare: „modelul a decodat adevărata dinamică neuronală”, „CfC/NODE este universal superior”, „activitatea DMFC converge spre puncte fixe” sau „un encoder este teoretic optim”. Pe Lorenz se poate afirma recuperare operațională doar pentru metricile cu adevăr cunoscut și alinierea declarată; pe NLB se poate afirma calitate predictivă și structură a reprezentării învățate.

## Bibliografie verificată
### Surse metodologice suplimentare

- Churchland, M. M. et al. (2012). *Neural population dynamics during reaching*. Nature, 487, 51--56. [DOI](https://doi.org/10.1038/nature11129).
- Gosztolai, A. et al. (2025). *Interpretable statistical representations of neural population dynamics and geometry*. Nature Methods. [DOI](https://doi.org/10.1038/s41592-024-02582-2), [arXiv](https://arxiv.org/abs/2304.03376v4), [codul MARBLE și instrucțiunile Linux](https://github.com/Dynamics-of-Neural-Systems-Lab/MARBLE).
- Owen, A. B. (2007). *The pigeonhole bootstrap*. The Annals of Applied Statistics, 1(2), 386--411. [DOI](https://doi.org/10.1214/07-AOAS122).
- Sussillo, D., Jozefowicz, R., Abbott, L. F. și Pandarinath, C. (2016). *LFADS - Latent Factor Analysis via Dynamical Systems*. [arXiv:1608.06315](https://arxiv.org/abs/1608.06315).
- Keshtkaran, M. R. și Pandarinath, C. (2019). *Enabling hyperparameter optimization in sequential autoencoders for spiking neural data*. [arXiv:1908.07896](https://arxiv.org/abs/1908.07896).
- Sedler, A. R. și Pandarinath, C. (2023). *lfads-torch: A modular and extensible implementation of latent factor analysis via dynamical systems*. [arXiv:2309.01230](https://arxiv.org/abs/2309.01230), [implementare și configurații NLB](https://github.com/arsedler9/lfads-torch).
- Neural Latents Benchmark. *NLB'21 status, public test data and submission policy*. [Anunț oficial privind închiderea EvalAI și publicarea testului în ianuarie 2026](https://neurallatents.github.io/), [cod și date de evaluare](https://github.com/neurallatents/nlb_tools).

1. Lorenz, E. N. (1963). *Deterministic nonperiodic flow*. Journal of the Atmospheric Sciences, 20(2), 130--141. [DOI](https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2).
2. Pandarinath, C. et al. (2018). *Inferring single-trial neural population dynamics using sequential auto-encoders*. Nature Methods, 15, 805--815. [DOI](https://doi.org/10.1038/s41592-018-0109-9).
3. Chen, R. T. Q., Rubanova, Y., Bettencourt, J. și Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS. [DOI/arXiv](https://doi.org/10.48550/arXiv.1806.07366).
4. Rubanova, Y., Chen, R. T. Q. și Duvenaud, D. (2019). *Latent Ordinary Differential Equations for Irregularly-Sampled Time Series*. NeurIPS 32. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html).
5. Hasani, R. et al. (2022). *Closed-form continuous-time neural networks*. Nature Machine Intelligence, 4, 992--1003. [DOI](https://doi.org/10.1038/s42256-022-00556-7).
6. Pei, F. et al. (2021). *Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity*. NeurIPS Datasets and Benchmarks. [DOI/arXiv](https://doi.org/10.48550/arXiv.2109.04463), [cod oficial](https://github.com/neurallatents/nlb_tools).
7. Facco, E., d'Errico, M., Rodriguez, A. și Laio, A. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*. Scientific Reports, 7, 12140. [DOI](https://doi.org/10.1038/s41598-017-11873-y).
8. Benjamini, Y. și Hochberg, Y. (1995). *Controlling the false discovery rate: a practical and powerful approach to multiple testing*. Journal of the Royal Statistical Society B, 57(1), 289--300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).