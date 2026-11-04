// Turns English into a list of signs the avatar knows. Runs in the browser, works offline.
// Only true synonyms are mapped. A medical word is never swapped for a different meaning
// (for example "fever" is not turned into "hot"); words without a sign are shown as text.
(() => {
  // Groups of true synonyms. Each sign language names its signs differently (ISL "mother",
  // ASL "mom"), so any word in a group finds whichever sign the chosen language has.
  const GROUPS = [
    ['hello', 'hi', 'hey', 'namaste'], ['thankyou', 'thanks'], ['sick', 'ill', 'unwell', 'sickness', 'illness'],
    ['medicine', 'tablet', 'pill', 'medication', 'drug', 'capsule'], ['doctor', 'physician', 'doc', 'dr'],
    ['bathroom', 'toilet', 'washroom', 'restroom', 'loo'], ['mother', 'mom', 'mum', 'mummy', 'mommy', 'amma'],
    ['father', 'dad', 'daddy', 'papa', 'appa'], ['grandmother', 'grandma', 'granny'], ['grandfather', 'grandpa'],
    ['child', 'kid'], ['alright', 'ok', 'okay', 'fine'], ['biglarge', 'big', 'large', 'huge'],
    ['smalllittle', 'small', 'little', 'tiny'], ['cellphone', 'phone', 'mobile'], ['house', 'home'],
    ['happy', 'glad'], ['sad', 'upset', 'unhappy'], ['fast', 'quick', 'quickly'], ['slow', 'slowly'],
    ['loud', 'loudly'], ['quiet', 'quietly'], ['storeorshop', 'shop', 'store'], ['streetorroad', 'road', 'street'],
    ['television', 'tv'], ['price', 'cost'], ['money', 'cash'],
    // ASL names (Google ISLR vocabulary)
    ['owie', 'hurt', 'hurts', 'pain', 'painful', 'ache', 'sore'], ['potty', 'toilet', 'bathroom'],
    ['minemy', 'my', 'mine'], ['weus', 'we', 'us'], ['hesheit', 'he', 'she', 'him', 'her'],
    ['kitty', 'cat'], ['glasswindow', 'window'],
  ];
  const GROUP_OF = new Map();
  GROUPS.forEach(g => g.forEach(w => GROUP_OF.set(w, [...(GROUP_OF.get(w) || []), ...g])));
  // One-way: possessives and object pronouns sign as the plain pronoun.
  const PRONOUN = { me: 'i', my: 'i', mine: 'i', myself: 'i', your: 'you', yours: 'you', yourself: 'you',
    his: 'he', him: 'he', her: 'she', hers: 'she', our: 'we', us: 'we', their: 'they', them: 'they' };
  const PHRASES = {
    'good morning': 'goodmorning', 'good afternoon': 'goodafternoon', 'good evening': 'goodevening',
    'good night': 'goodnight', 'thank you': 'thankyou', 'how are you': 'howareyou',
    'train station': 'trainstation', 'railway station': 'trainstation', 'train ticket': 'trainticket',
    't shirt': 'tshirt', 'you all': 'youplural', 'all of you': 'youplural',
    'have to': 'haveto', 'call on the phone': 'callonphone', 'phone call': 'callonphone',
    'ice cream': 'icecream', 'french fries': 'frenchfries',
  };
  // Words ISL leaves out; dropping them keeps signing short and clear.
  const DROP = new Set(('a an the is am are was were be been being to of for and or but will shall would ' +
    'should can could do does did please just so very really that this these those there here it\'s its ' +
    'with at in on by from as into about than then also any some').split(' '));

  const IRREGULAR = { men: 'man', women: 'woman', wives: 'wife', children: 'child', kids: 'child' };

  class SignPlanner {
    constructor(clips) {
      this.byLower = new Map(clips.map(c => [c.name.toLowerCase(), c.name]));
      this.label = new Map(clips.map(c => [c.name, c.label]));
    }

    find(w) { return this.byLower.get(w) || null; }

    lookup(w) {
      if (IRREGULAR[w]) w = IRREGULAR[w];
      const tries = [w, ...(GROUP_OF.get(w) || [])];
      if (PRONOUN[w]) tries.push(PRONOUN[w], ...(GROUP_OF.get(PRONOUN[w]) || []));
      for (const cand of tries) {
        const hit = this.find(cand);
        if (hit) return hit;
      }
      return null;
    }

    base(w) {
      const direct = this.lookup(w);
      if (direct) return direct;
      for (const [suf, rep] of [['ies', 'y'], ['es', ''], ['s', ''], ['ing', ''], ['ing', 'e'], ['ed', ''], ['ed', 'e'], ['ly', '']]) {
        if (w.endsWith(suf) && w.length > suf.length + 2) {
          const hit = this.lookup(w.slice(0, -suf.length) + rep);
          if (hit) return hit;
        }
      }
      return null;
    }

    /** -> [{word, sign|null}] in reading order. */
    plan(text) {
      const words = (text || '').toLowerCase().replace(/[’']/g, "'").replace(/[^a-z0-9' ]+/g, ' ')
        .replace(/-/g, ' ').split(/\s+/).filter(Boolean);
      const out = [];
      for (let i = 0; i < words.length; i++) {
        let hit = null;
        for (const n of [3, 2]) {
          const ph = words.slice(i, i + n).join(' ');
          if (words.length - i >= n && PHRASES[ph] && this.find(PHRASES[ph])) { hit = { n, sign: this.find(PHRASES[ph]) }; break; }
        }
        if (hit) { out.push({ word: words.slice(i, i + hit.n).join(' '), sign: hit.sign }); i += hit.n - 1; continue; }
        const w = words[i].replace(/'s$/, '').replace(/'/g, '');
        if (DROP.has(words[i]) || DROP.has(w)) continue;
        if (w === "don't" || w === 'dont' || w === 'not' || w === 'no' || w === 'never') { out.push({ word: 'not', sign: null }); continue; }
        out.push({ word: w, sign: this.base(w) });
      }
      return out;
    }
  }

  window.SignPlanner = SignPlanner;
})();
