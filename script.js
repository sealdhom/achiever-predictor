// ----------------------------
// QUESTIONS (same order as training)
// ----------------------------
const questions = [
  "On the whole, I am satisfied with myself.",
  "I feel that I have a number of good qualities.",
  "I am able to do things as well as most other people.",
  "I feel I do not have much to be proud of.",        // NEG
  "I certainly feel useless at times.",               // NEG
  "I take a positive attitude toward myself.",
  "I wish I could have more respect for myself.",     // NEG
  "All in all, I am inclined to feel that I am a failure.", // NEG
  "I feel that I’m a person of worth.",
  "I feel good about myself most of the time.",
  "I manage my study time effectively.",
  "I review my notes regularly even without exams.",
  "I am motivated to perform well in my studies.",
  "I set academic goals and work hard to achieve them.",
  "I often feel stressed because of academic pressure.", // NEG
  "I can manage my stress effectively.",
  "My classmates or friends encourage me to do my best in school.",
  "I study or collaborate with peers to understand lessons better.",
  "My family provides emotional or financial support for my studies.",
  "My parents or guardians express pride in my academic achievements.",
  "I get enough sleep (7–8 hours) during weekdays.",
  "I feel well-rested and alert during classes.",
  "I spend more than 4 hours a day on social media.", // NEG
  "My social media use affects my study time."        // NEG
];

// ----------------------------
// Reverse-scored item indices (0-based)
// ----------------------------
const reverseItems = [3, 4, 6, 7, 14, 22, 23];

// ----------------------------
// StandardScaler params
// ----------------------------
const means = [2.64,2.58,2.59,3.17,3.09,2.47,2.39,3.44,2.31,2.54,2.76,3.25,2.55,2.55,2.24,2.67,2.37,2.65,2.04,2.42,3.26,3.05,2.22,2.36];
const scales = [1.034601372510205,0.8852118390532291,0.9496841580230766,1.1229870880824944,1.2007914056987583,1.053138167573467,1.0573079021741965,1.1341957503006261,1.0648474069086145,0.9425497334358544,1.0111379727811631,0.9733961166965892,0.9420721840708386,1.0988630487917956,1.1586198686368194,0.970103087305674,1.0738249391777042,1.1608186766243902,1.1568923891183658,1.1329607230614838,1.1884443613396463,0.9937303457175894,1.1452510641776328,1.204325537385968];

// ----------------------------
// KMeans centroids
// ----------------------------
const centroids = [
  [-0.794333164109615,-0.7236754326413162,-0.6212591786601785,0.1724297328894481,0.2783517792150783,-0.6189294072591014,-0.28287966870943965,0.44030665717233713,-0.5756977939308762,-0.6372141854167053,-0.8415360849010958,-0.6304076553324038,-0.8733165957327613,-0.6935543177643176,0.002092353575029347,-0.659411332741594,-0.5421006636025603,-0.6643691532038163,-0.5846383317285659,-0.5044439227961978,-0.6012443474762247,-0.5992113937353474,-0.21855734356620296,0.3552849911391189],
  [0.1992592042031266,-0.017958905649566245,-0.027269525008551856,-0.01438471227788464,-0.20307120178317395,-0.1541181369218298,-0.2476051406184799,0.01899007427311615,-0.24296246301527794,0.05277555971742791,0.2627149140788104,0.296345479002412,0.12384047490133097,-0.05716864616531065,-0.4727109576816582,-0.003436060947492805,-0.1296587219676489,0.14689014223019256,-0.3005398868232456,-0.14439136349554257,0.36375930280857555,0.38833214418824347,-0.32643161446403113,-0.6395749208542733],
  [0.6586387661319768,0.8779173784820178,0.7701808703971226,-0.18318490737551024,-0.045208280162635346,0.9441170635536315,0.6782724839832928,-0.5453833065478088,1.0169129734755258,0.6774936174918422,0.6258860411665202,0.33021353374554474,0.8567738977867594,0.8970310602382009,0.6559528457717384,0.7819492984794583,0.8195004305579564,0.5784095181695873,1.107647161898188,0.7956397367357388,0.20194466632789249,0.16532222749732073,0.7122580464922101,0.47210775734734783]
];

const labelMap = {
  0: "High Achiever",
  1: "Neutral Achiever",
  2: "Low Achiever"
};

// ----------------------------
// Build form dynamically
// ----------------------------
const form = document.getElementById("quiz");

questions.forEach((q,i)=>{
  const div = document.createElement("div");
  div.className = "q";
  div.innerHTML = `
    <label for="q${i+1}">${i+1}. ${q}</label>
    <div class="scale">
      <span>1</span>
      <input type="range" id="q${i+1}" min="1" max="5" value="3">
      <span>5</span>
      <div class="val" id="v${i+1}">3</div>
    </div>`;
  form.appendChild(div);
});

// slider display updates
questions.forEach((_,i)=>{
  const slider = document.getElementById(`q${i+1}`);
  const val = document.getElementById(`v${i+1}`);
  slider.addEventListener("input",()=> val.textContent = slider.value);
});

// ----------------------------
// ML logic with reverse scoring
// ----------------------------
function scaleVector(arr){
  const z=[];
  for(let i=0;i<arr.length;i++){
    z.push((arr[i] - means[i]) / scales[i]);
  }
  return z;
}

function euclidean(a,b){
  let s=0;
  for(let i=0;i<a.length;i++){
    const d=a[i]-b[i];
    s+=d*d;
  }
  return Math.sqrt(s);
}

function softmaxFromDistances(dist){
  const alpha=1;
  const raw = dist.map(d => Math.exp(-alpha * d));
  const sum = raw.reduce((a,b)=>a+b,0);
  return raw.map(r=>r/sum);
}

function predictFromAnswers(raw){
  // APPLY REVERSE SCORING
  const answers = raw.map((v,i)=>{
    return reverseItems.includes(i) ? (6 - v) : v;
  });

  const z = scaleVector(answers);
  const distances = centroids.map(c => euclidean(z,c));
  const probs = softmaxFromDistances(distances);

  let best = 0;
  for(let i=1;i<probs.length;i++){
    if(probs[i] > probs[best]) best = i;
  }

  return {
    cluster: best,
    label: labelMap[best],
    probs,
    distances
  };
}

// ----------------------------
// Interpretation text
// ----------------------------
function interpretation(label){
  if(label === "High Achiever")
    return "Your responses align closely with students who show strong study habits, motivation, and self-efficacy.";
  if(label === "Neutral Achiever")
    return "You show a balanced set of strengths and areas for growth.";
  return "Your responses resemble patterns of students who may struggle with consistency or support.";
}

// ----------------------------
// Show results to user
// ----------------------------
function showResults({label,probs,distances}){
  const out = document.getElementById("output");
  out.style.display="block";
  out.innerHTML="";

  const card=document.createElement("div");
  card.className="result-card";

  const header=document.createElement("div");
  header.className="summary";
  header.innerHTML=`
    <div>
      <div class="prediction-title">Prediction Result:</div>
      <div class="tag">${label}</div>
    </div>
    <div>
      <div class="interpretation-title">Interpretation</div>
      <div>${interpretation(label)}</div>
    </div>`;
  card.appendChild(header);

  // Probability bars
  [
    {label:"High Achiever", p:probs[0]},
    {label:"Neutral Achiever", p:probs[1]},
    {label:"Low Achiever", p:probs[2]}
  ].forEach(item=>{
    const row=document.createElement("div");
    row.style.display="grid";
    row.style.gridTemplateColumns="1fr 70px";
    row.style.gap="10px";
    row.style.marginTop="12px";

    row.innerHTML=`
      <div>
        <div style="font-weight:700">${item.label}</div>
        <div class="bar"><i style="width:${(item.p*100).toFixed(1)}%"></i></div>
      </div>
      <div style="text-align:right;font-weight:800">${(item.p*100).toFixed(1)}%</div>
    `;

    card.appendChild(row);
  });

  out.appendChild(card);
}

// ----------------------------
// Events
// ----------------------------
document.getElementById("predict").addEventListener("click",e=>{
  e.preventDefault();

  const raw=[];
  for(let i=1;i<=questions.length;i++){
    raw.push(parseInt(document.getElementById("q"+i).value));
  }

  const result = predictFromAnswers(raw);
  showResults(result);
});

document.getElementById("reset").addEventListener("click",()=>{
  for(let i=1;i<=questions.length;i++){
    const el=document.getElementById("q"+i);
    el.value=3;
    document.getElementById("v"+i).textContent="3";
  }
  document.getElementById("output").style.display="none";
});
