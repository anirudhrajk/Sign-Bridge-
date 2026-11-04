// Turns English into a list of signs the avatar knows. Runs in the browser, works offline.
// Only true synonyms are mapped. A medical word is never swapped for a different meaning
// (for example "fever" is not turned into "hot"); words without a sign are shown as text.
(() => {
  const SYNONYMS = {
    hi: 'hello', hey: 'hello', namaste: 'hello', thanks: 'thankyou',
    ill: 'sick', unwell: 'sick', sickness: 'sick', illness: 'sick',
    tablet: 'medicine', tablets: 'medicine', pill: 'medicine', pills: 'medicine', medication: 'medicine',
    medications: 'medicine', medicines: 'medicine', capsule: 'medicine', capsules: 'medicine', drug: 'medicine', drugs: 'medicine',
    physician: 'doctor', doc: 'doctor', dr: 'doctor', doctors: 'doctor',
    toilet: 'bathroom', washroom: 'bathroom', restroom: 'bathroom', loo: 'bathroom',
    mom: 'mother', mum: 'mother', mummy: 'mother', mommy: 'mother', amma: 'mother',
    dad: 'father', daddy: 'father', papa: 'father', appa: 'father',
    kid: 'child', kids: 'child', children: 'child', baby: 'baby',
    ok: 'alright', okay: 'alright', fine: 'alright',
    large: 'biglarge', big: 'biglarge', huge: 'biglarge', small: 'smalllittle', little: 'smalllittle', tiny: 'smalllittle',
    mobile: 'cellphone', phone: 'cellphone', cellphone: 'cellphone',
    home: 'house', glad: 'happy', upset: 'sad', unhappy: 'sad',
    quick: 'fast', quickly: 'fast', slowly: 'slow', loudly: 'loud', quietly: 'quiet',
    shop: 'storeorshop', store: 'storeorshop', road: 'streetorroad', street: 'streetorroad',
    tv: 'television', cost: 'price', cash: 'money',
    me: 'i', my: 'i', mine: 'i', myself: 'i', your: 'you', yours: 'you', yourself: 'you',
    his: 'he', him: 'he', her: 'she', hers: 'she', our: 'we', us: 'we', their: 'they', them: 'they',
  };
  const PHRASES = {
    'good morning': 'goodmorning', 'good afternoon': 'goodafternoon', 'good evening': 'goodevening',
    'good night': 'goodnight', 'thank you': 'thankyou', 'how are you': 'howareyou',
    'train station': 'trainstation', 'railway station': 'trainstation', 'train ticket': 'trainticket',
    't shirt': 'tshirt', 'you all': 'youplural', 'all of you': 'youplural',
  };
  // Words ISL leaves out; dropping them keeps signing short and clear.
  const DROP = new Set(('a an the is am are was were be been being to of for and or but will shall would ' +
    'should can could do does did please just so very really that this these those there here it\'s its ' +
    'with at in on by from as into about than then also any some').split(' '));

  const IRREGULAR = { men: 'man', women: 'woman', wives: 'wife', children: 'child' };

  class SignPlanner {
    constructor(clips) {
      this.names = new Set(clips.map(c => c.name));
      this.label = new Map(clips.map(c => [c.name, c.label]));
    }

    base(w) {
      if (this.names.has(w)) return w;
      if (SYNONYMS[w] && this.names.has(SYNONYMS[w])) return SYNONYMS[w];
      if (IRREGULAR[w] && this.names.has(IRREGULAR[w])) return IRREGULAR[w];
      for (const [suf, rep] of [['ies', 'y'], ['es', ''], ['s', ''], ['ing', ''], ['ing', 'e'], ['ed', ''], ['ed', 'e'], ['ly', '']]) {
        if (w.endsWith(suf) && w.length > suf.length + 2) {
          const b = w.slice(0, -suf.length) + rep;
          if (this.names.has(b)) return b;
          if (SYNONYMS[b] && this.names.has(SYNONYMS[b])) return SYNONYMS[b];
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
          if (words.length - i >= n && PHRASES[ph] && this.names.has(PHRASES[ph])) { hit = { n, sign: PHRASES[ph] }; break; }
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
