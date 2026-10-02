# Plan științific revizuit

## Titlu propus

**Modelarea dinamicii latente a activității neuronale: comparație controlată între un VAE secvențial cu GRU, Latent Neural ODE și Latent CfC, cu evaluare NLB și analiză geometrică.**

În implementarea descrisă mai jos, primul model se va numi **GRU-LVAE** (variational autoencoder latent cu generator GRU), nu „LFADS”. Denumirea LFADS se poate folosi numai dacă se implementează explicit arhitectura publicată: factori latenți, condiție inițială per trial și intrări inferate de controller. LFADS publicat inferă atât condiții inițiale, cât și intrări; un BiGRU urmat numai de un GRU generator nu este echivalent cu LFADS [Sussillo et al., 2018](https://doi.org/10.1038/s41592-018-0109-9).

## 1. Scop și întrebări de cercetare

Scopul este evaluarea empirică a trei **modele generative ale spike-counturilor binned**, cu același model de observație Poisson și același protocol de selecție. Nu se va pretinde că niciun model recuperează în mod unic „adevărata” stare neuronală: reprezentările latente ale unui VAE sunt neidentificabile până la transformări ale coordonatelor. Rezultatele despre geometrie vor fi formulate ca proprietăți ale reprezentării învățate, nu ca proprietăți demonstrate ale creierului.

### Întrebări primare, pre-înregistrate

1. La capacitate aproximativ egală, diferă modelele în co-smoothing pe setul NLB de test?
2. Pentru `MC_Maze`, diferă performanța oficială de decodare a vitezei mâinii obținută din ratele inferate?
3. Pe un set sintetic cu dinamică Lorenz și observații neregulate controlate, aduce un encoder sensibil la intervalele temporale un avantaj față de un encoder discret?

### Întrebări secundare, exploratorii

1. Cum diferă dimensiunea efectivă, continuitatea temporală și separabilitatea condițiilor în reprezentările latente?
2. Ce proprietăți locale ale hărților dinamice învățate sunt stabile între seed-uri?
3. Care este compromisul între calitate, număr de parametri, număr de evaluări ale dinamicii și timp de execuție?

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

Toate modelele primesc spike-counturi $x_{1:T}\in\mathbb{N}^{N}$ la binuri de $\Delta t=5$ ms, produc rate $r_{1:T}>0$ și folosesc aceeași distribuție observațională:

$$
x_{t,n}\sim \operatorname{Poisson}(r_{t,n}\Delta t),\qquad
r_t=\exp(W_{\mathrm{out}}z_t+b_{\mathrm{out}}).
$$

În cod se folosește stabilizare numerică a log-ratei înainte de exponențială, iar ratele sunt raportate în Hz. Decoderul, dimensiunea latentă, distribuția priorului, obiectivul, procesarea datelor și numărul țintă de parametri sunt aceleași pentru toate modelele.

### M0 — GRU-LVAE, baseline discret

- Posterior: encoder BiGRU care produce $\mu$ și $\log\sigma^2$ pentru $q(z_0\mid x_{1:T})$.
- Stare inițială: $z_0=\mu+\sigma\odot\epsilon$, cu $\epsilon\sim\mathcal{N}(0,I)$.
- Dinamică autonomă discretă: $z_{t+1}=F_{\mathrm{GRU}}(z_t)$, fără datele observate ca intrare în generator după $t=0$.
- Citire Poisson: comună tuturor modelelor.

Aceasta este comparația controlată a unui generator GRU cu celelalte blocuri dinamice. O implementare LFADS completă poate fi inclusă ulterior ca **baseline separat**, cu controller și inputuri inferate; nu trebuie amestecată cu testul controlat, fiindcă schimbă modelul generativ.

### M1 — Latent Neural ODE

- Posterior de bază: același BiGRU ca M0, pentru comparația de dinamică.
- Dinamică autonomă: $\dot z(t)=f_\theta(z(t))$, cu $f_\theta$ MLP Tanh. Timpul $t$ nu este intrare în modelul primar, pentru ca punctele fixe să fie bine definite.
- Integrare la instanțele de bin: solver adaptiv explicit, toleranțe predefinite și logarea numărului de function evaluations (NFE).

Un model condiționat de timp sau de covariate se va trata separat. Pentru $\dot z=f(z,t)$, un punct fix al lui $f(z)$ nu are în general interpretare autonomă; analiza corectă necesită un sistem augmentat sau un timp înghețat, declarat explicit.

### M2 — Latent CfC

- Posterior de bază: același BiGRU ca M0 și M1.
- Dinamică: $z_{t+1}=F_{\mathrm{CfC},\Delta t}(z_t)$, cu intrare externă nulă după inițializare și `timespans=\Delta t` transmisă explicit conform API-ului versiunii fixate a `ncps`.
- Citire Poisson: comună.

CfC este tratat ca model cu pas temporal variabil, nu ca echivalent direct al unui Neural ODE. Reproducerea cere versiunea exactă a `ncps` și un test unitar care verifică faptul că schimbarea `timespans` schimbă ieșirea celulei.

### Paritate de capacitate

Înainte de experimentele principale, se alege pentru fiecare bloc dinamic o configurație a cărei număr total de parametri este în interiorul a ±10% față de celelalte două. Se raportează separat parametrii encoderului, dinamicii, decoderului și totalul. Dacă paritatea nu este realizabilă, se fac două analize: una cu capacitate potrivită și una cu configurațiile recomandate de autori; aceste două regimuri nu se combină statistic.

## 4. Ablația encoderului, formulată corect

Ablația are două regimuri, fiecare cu dinamică, decoder și buget de parametri fixate.

| Regim | M0 | M1 | M2 | Interpretare admisă |
|---|---|---|---|---|
| C: encoder comun | BiGRU | BiGRU | BiGRU | Efect asociat cu alegerea blocului dinamic, condiționat de arhitectura și capacitatea fixate. |
| E: encoder alternativ | BiGRU | ODE-RNN | CfC bidirecțional | Efect empiric al înlocuirii encoderului pentru acel model; nu „optimalitate” și nu cauzalitate biologică. |

Pentru M1 și M2 se face contrastul pereche C versus E cu **aceleași seed-uri**, aceeași ordine de mini-batch-uri, același set de hiperparametri selectați și aceeași dinamică generativă. Astfel, singura schimbare intenționată este encoderul. Se raportează și diferența de parametri; dacă aceasta depășește 10%, se execută o variantă de capacitate potrivită.

Pe NLB, unde binurile sunt regulate, această ablație nu testează un avantaj al timestampurilor neregulate. Ea testează doar utilitatea inductive bias-ului ODE-RNN/CfC la date regulate. Avantajul pentru timp neregulat se testează exclusiv în experimentul sintetic din secțiunea 5.

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
- **L-neregulat**: se păstrează aceeași traiectorie latentă și aceleași observații Poisson, dar se observă la o submulțime temporală aleasă printr-un mecanism MCAR fixat înainte (de exemplu, intervale între observații trase dintr-o distribuție geometrică, cu aceeași rată de observare pentru toate modelele). Se transmite masca și vectorul de timp. Testul principal pentru encoderul ODE-RNN este diferența C versus E pe acest set.

Validarea generatoarelor Lorenz este în două etape. Mai întâi algoritmul QR este rulat asupra sistemului Lorenz analitic până la convergență; rezultatul, pasul de integrare, timpul de burn-in și timpul total sunt salvate. Apoi, pentru dinamica Neural ODE se estimează spectrul pe traiectorii lungi ale modelului. Criteriul minim este un interval de încredere al lui $\lambda_1$ complet pozitiv, nu egalitatea exactă cu o singură cifră publicată. O eroare relativă față de estimatorul de referință este raportată, dar nu este optimizată în timpul antrenării.

Pentru compararea geometriei cu adevărul Lorenz, latentele sunt aliniate post-hoc cu o mapare afină/Procrustes estimată **numai pe train**, apoi evaluată pe test. Se raportează eroarea de traiectorie aliniată, corelația distanțelor pereche și semnul lui $\lambda_1$. Fără această aliniere, coordonatele brute ale VAE nu au o corespondență identificabilă cu cele trei coordonate Lorenz.

### 5.2 NLB `MC_Maze`

- Se utilizează API-ul și spliturile publicate în versiunea fixată a `nlb_tools`; nu se reconstruiesc splituri proprii.
- `MC_Maze-S` are 100 trialuri train, `MC_Maze-M` 250 trialuri train, iar fiecare are 100 trialuri test; ele provin din sesiuni diferite, nu sunt subseturi imbricate. Nu se transferă hiperparametri aleși pe M către rezultatul S sau invers fără a declara transferul [Pei et al., 2021, Tabelul 4](https://arxiv.org/html/2109.04463v4).
- Metrica primară este co-smoothing bps pe neuroni held-out. Pentru inferența de test se folosesc numai neuronii held-in disponibili.
- Metricile NLB secundare sunt: $R^2$ pentru viteza mâinii prin decoder Ridge conform protocolului NLB, PSTH-$R^2$ și forward-prediction bps acolo unde este aplicabil. Forward prediction este raportată ca metrică secundară, fiind justificată pentru activitate previzibilă, nu ca probă că sistemul biologic este autonom.

Mișcarea poate avea structură rotațională în anumite proiecții, dar nu se va postula a priori existența unui ciclu limită în dinamica fiecărui model; aceasta este o ipoteză exploratorie.

### 5.3 NLB `DMFC_RSG`

- Se utilizează spliturile NLB: 1006 trialuri train, 283 test, 40 neuroni held-in și 14 held-out [Pei et al., 2021, Tabelul 7](https://arxiv.org/html/2109.04463v4).
- Primar: co-smoothing bps. Secundar oficial: corelația Pearson dintre viteza neuronală medie pe trial și $t_p$.
- PSTH-$R^2$ se calculează cu protocolul NLB. O regresie directă din latent spre $t_p$ este o analiză suplimentară și nu este confundată cu metrica benchmarkului.

Nu se presupune „rampă spre puncte fixe”. DMFC_RSG implică atât structură internă, cât și influență de sarcină; o dinamică autonomă poate fi o aproximație utilă, nu un model demonstrat al circuitului [Pei et al., 2021](https://neurallatents.github.io/datasets).

## 6. Antrenare și selecție fără leakage

Obiectivul fiecărui model este ELBO pe datele de antrenare:

$$
\mathcal{L}=\mathbb{E}_{q_\phi(z_0\mid x)}[\log p_\theta(x\mid z_{1:T})]
-\beta\,D_{KL}\!\left(q_\phi(z_0\mid x)\Vert\mathcal{N}(0,I)\right).
$$

- Selecția checkpointului și early stopping folosesc exclusiv validation co-bps.
- Testul se accesează o singură dată pentru fiecare run final; după publicarea datelor NLB test în 2026, se păstrează un test intern blocat pentru alegerea de implementare și toleranțe.
- Hiperparametrii sunt aleși pe Lorenz train/validation și pe validation NLB, nu pe test. `MC_Maze-S` este pilot, nu set de validare pentru `MC_Maze-M`, deoarece sunt sesiuni distincte.
- Toate modelele au aceeași rată de învățare și aceeași schemă de annealing numai după o verificare pilot că nu apare posterior collapse. Dacă warm-up-ul este 50 epoci, early stopping nu poate opri un run înainte de epoca 50; după aceea se aplică patience declarată.
- Se fixează și se salvează: seed Python/NumPy/PyTorch, versiuni pachete și CUDA, hardware, precizie, batch size, masca de neuroni, configurația YAML, hash-ul commitului, numărul de parametri, NFE și timpul wall-clock.

Solverul Neural ODE este parte a modelului efectiv. Configurația primară poate fi `dopri5`, `rtol=1e-3`, `atol=1e-4`, însă `dopri5`, `rk4` și `euler` sunt comparate în ablația solverului cu aceleași setări în rest. Metoda adjoint și metoda directă nu se tratează ca identice: se loghează metoda de gradient, stabilitatea și memoria. Neural ODE permite explicit compromisul precizie-viteză prin solver [Chen et al., 2018](https://doi.org/10.48550/arXiv.1806.07366).

## 7. Metrici și reguli de calcul

### 7.1 Endpointuri primare

1. co-bps NLB pentru fiecare dataset;
2. $R^2$ velocity NLB pentru `MC_Maze`;
3. diferența C versus E pentru M1 pe L-neregulat, măsurată prin held-out Poisson log-likelihood/bps;
4. latența/timpul de antrenare și NFE, raportate descriptiv, nu transformate într-un scor compozit arbitrar.

### 7.2 Metrici secundare

- PSTH-$R^2$ și forward-prediction bps exact ca în `nlb_tools`.
- Participation ratio $\mathrm{PR}=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$, calculat din covarianța latentelor centrate. Se raportează atât PR, cât și $\mathrm{PR}/d$; comparația se face la același $d$.
- TwoNN intrinsic dimension, cu aceleași reguli de subsampling și whitening pentru toate modelele. Timepointurile autocorelate nu sunt tratate eronat ca eșantioane independente: se subsamplează uniform pe trialuri și se folosesc bootstrap-uri pe trialuri. Estimatorul TwoNN este definit de [Facco et al., 2017](https://doi.org/10.1038/s41598-017-11873-y).
- Smoothness normalizat: media lui $\lVert\tilde z_{t+1}-\tilde z_t\rVert_2/\Delta t$, după standardizarea $\tilde z$ pe train. Fără normalizare, o simplă rescalare a latentelor schimbă metrica.
- Consistență intra-condiție: distanța medie față de media condiției după aliniere temporală explicită; se raportează numai pentru condiții cu număr suficient de trialuri.
- Separabilitate: balanced accuracy a unui LDA antrenat pe train, cu regularizare aleasă pe validation și evaluat o dată pe test. Se raportează și baseline-ul de clasă majoritară.

MARBLE nu este endpoint obligatoriu. Se include numai ca analiză exploratorie dacă mediul și versiunea oficială pot fi reproduse într-un container Linux fixat; în caz contrar, se raportează transparent ca neexecutată, fără înlocuirea tacită cu o altă metrică. Nu se utilizează pentru concluzia primară.

## 8. Analiză dinamică compatibilă cu fiecare arhitectură

Toată analiza se aplică checkpointului selectat pe validation și se repetă pentru fiecare seed, nu doar pentru „cel mai bun” seed. Se lucrează în spațiul latent complet; graficele PCA/UMAP sunt numai vizualizări și nu sunt folosite pentru calculul Jacobianului.

| Analiză | GRU-LVAE | Neural ODE autonom | CfC ca hartă $F_{\Delta t}$ |
|---|---:|---:|---:|
| Punct fix | $F(z^*)=z^*$ | $f(z^*)=0$ | $F_{\Delta t}(z^*)=z^*$ |
| Stabilitate locală | multiplicatori ai $\partial F/\partial z$; stabil dacă $|\mu_i|<1$ | valori proprii ale $\partial f/\partial z$; stabil dacă $\operatorname{Re}(\lambda_i)<0$ | multiplicatori ai $\partial F_{\Delta t}/\partial z$ |
| Exponenți Lyapunov | produs QR al Jacobienilor hărții, pe pas | ecuație variațională/QR, pe unitatea de timp | produs QR al Jacobienilor hărții |
| Divergență, nulcline, Floquet continuu | nu | da, numai pentru $f(z)$ autonom | nu, decât dacă se implementează și validează separat un vector field echivalent |

### Reguli pentru puncte fixe

- Se pornește optimizarea din stări latente observate și din puncte de referință aleatoare; numărul de porniri, toleranța reziduală și regula de deduplicare sunt declarate înainte.
- Pentru o hartă discretă se minimizează $\lVert F(z)-z\rVert^2$, nu $\lVert f(z)\rVert^2$.
- Un punct este raportat doar dacă reziduul normalizat este sub toleranța declarată și clasificarea sa este stabilă în cel puțin o proporție predefinită de seed-uri.

### Reguli pentru exponenții Lyapunov

Se raportează unitatea ($\mathrm{s}^{-1}$ pentru ODE, `per step` și conversie la $\mathrm{s}^{-1}$ declarată pentru hărți), durata totală, transientul eliminat, frecvența QR și intervalul bootstrap 95% pe traiectorii. Spectrele provenite din hărți și ODE nu sunt prezentate ca numeric identice; sunt comparate numai după conversia de unitate și cu această limitare în legendă.

Pentru Lorenz, se verifică estimatorul pe sistemul adevărat înainte de analiza modelelor. Pentru datele NLB, un exponent pozitiv sau negativ este o proprietate a generatorului ajustat, nu o demonstrație a haosului ori a disipării dinamiciilor cerebrale reale.

### Cicluri și bifurcații

Detectarea unui ciclu limită este exploratorie și cere: perioadă estimată pe traiectorii generate liber, eroare de închidere sub prag declarat și robustețe la seed-uri. O periodicitate în date sau în proiecția PCA nu este suficientă.

Nu se va numi „bifurcație” simpla scalare $f_\alpha=\alpha f$. Pentru un ODE autonom, această scalare pozitivă modifică în principal parametrizarea temporală și nu schimbă punctele fixe; prin urmare nu constituie un control fizic care poate identifica în general saddle-node, Hopf sau pitchfork. O analiză de bifurcație este permisă numai dacă se introduce un parametru de control cu sens experimental sau generativ explicit (de exemplu o covariată de condiție), se antrenează/validează un model condiționat și se analizează familia $f(z;c)$; altfel secțiunea este eliminată.

## 9. Plan statistic

- Zece seed-uri independente sunt minimul pentru endpointurile primare. Pilotul poate avea trei seed-uri, dar nu generează afirmații de semnificație.
- Pentru fiecare contrast pre-înregistrat se raportează: valoare per seed, medie a diferențelor, deviație standard, interval bootstrap 95% pe trialuri/test și $p$ din test de permutare pereche sau model ierarhic declarat.
- Familiile de teste primare sunt ajustate separat prin Benjamini-Hochberg la $q=0.05$. Metricile exploratorii sunt etichetate exploratorii, cu intervale de încredere și fără limbaj confirmator.
- Co-bps se compară în cadrul aceluiași dataset; valorile absolute nu se compară între dataseturi cu populații și rate diferite, conform avertismentului NLB.
- Efectul este interpretat prin mărime și interval, nu exclusiv prin pragul $p<0.05$. Pragul practic de interes pentru co-bps, dacă este folosit, se declară înainte și nu înlocuiește intervalul de încredere.

## 10. Ordinea experimentelor și criterii de trecere

1. **Teste unitare și Lorenz de referință.** Se verifică dimensiuni, likelihood Poisson, reproducibilitate, transmiterea $\Delta t$, paritatea parametrilor și estimatorul QR pe Lorenz analitic. Nu se trece mai departe dacă estimatorul de referință nu converge.
2. **L-regulat și L-neregulat.** Se confirmă ratele 5--20 Hz, separarea pe trialuri, co-smoothing/reconstrucția held-out și semnul pozitiv al $\lambda_1$ acolo unde analiza este aplicabilă. Rezultatul C versus E pe L-neregulat este verificarea directă a ipotezei despre gaps temporale.
3. **Pilot `MC_Maze-S`, trei seed-uri.** Se verifică compatibilitatea cu formatul NLB, masca held-in/held-out, lipsa NaN, KL posterior, NFE și costul memoriei. Nu se publică inferență statistică din pilot.
4. **Experimente confirmatorii.** Se rulează M0--M2 în regimul C și M1--M2 în regimul E, cu zece seed-uri, pe `MC_Maze-M` și `DMFC_RSG`; checkpointul vine numai din validation.
5. **Ablații declarate.** Se fac după rezultatul principal: capacitate potrivită versus configurație recomandată, solver NODE, dimensiune latentă și encoder C versus E. Sweep-ul de dimensiune nu schimbă retrospectiv rezultatul principal la $d=16$.
6. **Analize exploratorii.** Geometrie, analiză locală a hărților, vizualizări și, dacă mediul este fixat, MARBLE. Concluziile confirmatorii nu depind de ele.

## 11. Output minim și raportare

1. Un tabel pentru fiecare dataset și regim, cu seed, config hash, număr parametri, co-bps, metrici NLB aplicabile, timp/epocă, memorie GPU, NFE și checkpoint selectat.
2. Un tabel de contraste C versus E pentru M1 și M2: diferență, interval 95%, $p$ brut, $p$ FDR, cost suplimentar și parametri suplimentari.
3. Un tabel separat de analiză dinamică, cu tipul corect de sistem (`map`/`ODE`), unitatea exponentului, reziduurile punctelor fixe și intervalele de încredere. Celulele neaplicabile rămân `N/A`, nu `0`.
4. Curbe ELBO, co-bps și KL pentru fiecare seed; graficele agregate arată media și intervalul, plus punctele individuale.
5. Figuri de traiectorii, PCA/jPCA/UMAP numai cu seed, split, normalizare și metodă de aliniere în legendă. UMAP nu este folosit ca probă cantitativă de separare.
6. Un document `conclusions.md` care separă: rezultate primare confirmatorii, rezultate exploratorii, limitări, deviații de la protocol și afirmații care **nu** pot fi trase.

## 12. Afirmații finale permise și nepermise

Sunt permise formulări precum: „în acest protocol, cu această capacitate și aceste splituri, modelul M1 a avut co-bps mai mare decât M0 cu intervalul de încredere raportat” sau „encoderul ODE-RNN a îmbunătățit/nu a îmbunătățit performanța pe observații Lorenz neregulate simulate”.

Nu sunt permise fără dovezi suplimentare: „modelul a decodat adevărata dinamică neuronală”, „CfC/NODE este universal superior”, „activitatea DMFC converge spre puncte fixe” sau „un encoder este teoretic optim”. Pe Lorenz se poate afirma recuperare operațională doar pentru metricile cu adevăr cunoscut și alinierea declarată; pe NLB se poate afirma calitate predictivă și structură a reprezentării învățate.

## Bibliografie verificată

1. Lorenz, E. N. (1963). *Deterministic nonperiodic flow*. Journal of the Atmospheric Sciences, 20(2), 130--141. [DOI](https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2).
2. Sussillo, D., Jozefowicz, R., Abbott, L. F. și Pandarinath, C. (2018). *LFADS—Latent Factor Analysis via Dynamical Systems*. Nature Methods, 15, 805--815. [DOI](https://doi.org/10.1038/s41592-018-0109-9).
3. Chen, R. T. Q., Rubanova, Y., Bettencourt, J. și Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS. [DOI/arXiv](https://doi.org/10.48550/arXiv.1806.07366).
4. Rubanova, Y., Chen, R. T. Q. și Duvenaud, D. (2019). *Latent Ordinary Differential Equations for Irregularly-Sampled Time Series*. NeurIPS 32. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html).
5. Hasani, R. et al. (2022). *Closed-form continuous-time neural networks*. Nature Machine Intelligence, 4, 992--1003. [DOI](https://doi.org/10.1038/s42256-022-00556-7).
6. Pei, F. et al. (2021). *Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity*. NeurIPS Datasets and Benchmarks. [DOI/arXiv](https://doi.org/10.48550/arXiv.2109.04463), [cod oficial](https://github.com/neurallatents/nlb_tools).
7. Facco, E., d'Errico, M., Rodriguez, A. și Laio, A. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*. Scientific Reports, 7, 12140. [DOI](https://doi.org/10.1038/s41598-017-11873-y).
8. Benjamini, Y. și Hochberg, Y. (1995). *Controlling the false discovery rate: a practical and powerful approach to multiple testing*. Journal of the Royal Statistical Society B, 57(1), 289--300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).