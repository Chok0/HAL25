/*
 * jam.scd — moteur de synthese du jam generatif.
 *
 * Recoit l'etat musical envoye par Python (voir halj/bridge/osc.py) sur le
 * port 57120 (sclang) et produit :
 *   - un drone dark accorde sur la tonalite detectee, avec glissando ;
 *   - une couche kick / percu declenchee pas a pas par le sequenceur Python.
 *
 * Usage : ouvrir dans l'IDE SuperCollider et evaluer le bloc entier (Ctrl-Entree),
 * puis lancer cote Python :  halj mock        (reglage sans guitare)
 *                            halj jam         (chaine complete)
 */

s.options.sampleRate = 44100;
// Blocs courts : la latence de sortie compte dans le budget < 20-30 ms vise.
s.options.blockSize = 64;
s.options.hardwareBufferSize = 128;

s.waitForBoot({
    var drone, oscDefs, root, ampBus, chordDriven;

    // ---------------------------------------------------------------
    // SynthDefs
    // ---------------------------------------------------------------

    // Drone : sub sinus + nappe de dents de scie desaccordees, filtre passe-bas
    // module lentement. `lag` porte le glissando entre deux tonalites : le
    // changement de tonalite doit s'entendre comme une derive, pas comme un saut.
    //
    // `third` et `fifth` portent les deux autres notes de l'accord envoye par
    // /jam/chord, et `chordAmp` les ouvre. A zero — c'est-a-dire tant qu'aucun
    // accord n'est arrive — le drone est exactement celui d'avant : une seule
    // note, la tonique.
    SynthDef(\jamDrone, { |freq = 110, third = 110, fifth = 110, chordAmp = 0,
                          amp = 0.35, glide = 1.8, gate = 1, cutoff = 900|
        var f, f3, f5, sub, saws, chord, body, env, mod;
        f = Lag.kr(freq, glide);
        f3 = Lag.kr(third, glide);
        f5 = Lag.kr(fifth, glide);
        sub = SinOsc.ar(f / 2, 0, 0.6) + SinOsc.ar(f, 0, 0.35);
        // Trois dents de scie legerement desaccordees : le battement lent qui
        // donne son epaisseur au drone.
        saws = Mix.ar([
            Saw.ar(f * 1.001, 0.20),
            Saw.ar(f * 0.999, 0.20),
            Saw.ar(f * 2.002, 0.10)
        ]);
        // Tierce et quinte en retrait : elles doivent colorer l'accord, pas
        // concurrencer la fondamentale qui porte la basse.
        chord = Mix.ar([
            Saw.ar(f5 * 1.001, 0.10), SinOsc.ar(f5, 0, 0.10),
            Saw.ar(f3 * 0.999, 0.07), SinOsc.ar(f3, 0, 0.07)
        ]) * Lag.kr(chordAmp, glide);
        mod = SinOsc.kr(0.07).range(0.75, 1.35);
        body = LPF.ar(sub + saws + chord, Lag.kr(cutoff, 1.5) * mod);
        body = body + (0.18 * body.distort);
        env = EnvGen.kr(Env.asr(4.0, 1.0, 3.0), gate, doneAction: 2);
        // Reverb longue : la queue tient l'espace entre deux phrases de guitare.
        body = body + (0.25 * FreeVerb.ar(body, mix: 1, room: 0.92, damp: 0.4));
        Out.ar(0, Pan2.ar(body * env * Lag.kr(amp, 0.5), 0));
    }).add;

    // Kick : enveloppe de hauteur descendante, corps sinus.
    SynthDef(\jamKick, { |velocity = 0.8, freq = 52|
        var pitchEnv, ampEnv, click, body;
        pitchEnv = EnvGen.kr(Env([freq * 4.5, freq, freq * 0.9], [0.014, 0.09], \exp));
        ampEnv = EnvGen.kr(Env.perc(0.001, 0.34, velocity, -5), doneAction: 2);
        body = SinOsc.ar(pitchEnv);
        click = HPF.ar(WhiteNoise.ar(0.25), 1800)
            * EnvGen.kr(Env.perc(0.0005, 0.012));
        Out.ar(0, Pan2.ar((body + click) * ampEnv * 0.9, 0));
    }).add;

    // Percu : burst de bruit filtre, panoramique legerement aleatoire pour que
    // la grille euclidienne ne sonne pas mecanique.
    SynthDef(\jamPerc, { |velocity = 0.6, pan = 0.0, tone = 3200|
        var noise, env;
        env = EnvGen.kr(Env.perc(0.001, 0.085, velocity, -8), doneAction: 2);
        noise = BPF.ar(WhiteNoise.ar, tone, 0.6) + (0.3 * HPF.ar(WhiteNoise.ar, 6000));
        Out.ar(0, Pan2.ar(noise * env * 0.5, pan));
    }).add;

    // Marqueur d'attaque guitare : bref eclat harmonise avec le drone. Rend
    // audible le fait que l'analyse "entend" le jeu, sans surcharger le mix.
    SynthDef(\jamSpark, { |freq = 440, velocity = 0.3|
        var env, sig;
        env = EnvGen.kr(Env.perc(0.004, 0.5, velocity * 0.12, -6), doneAction: 2);
        sig = SinOsc.ar(freq * 4) + (0.4 * SinOsc.ar(freq * 6.01));
        Out.ar(0, Pan2.ar(sig * env, Rand(-0.4, 0.4)));
    }).add;

    s.sync;

    // ---------------------------------------------------------------
    // Etat + reception OSC
    // ---------------------------------------------------------------

    root = 110.0;
    drone = nil;
    // Passe a true des le premier /jam/chord : a partir de la, c'est la grille
    // d'accords qui pilote la hauteur du drone, et /jam/key ne fait plus que
    // colorer le timbre.
    chordDriven = false;

    // /jam/drone <amp> <glide> : demarre (ou reconfigure) le drone.
    OSCdef(\jamDrone, { |msg|
        var amp = msg[1] ? 0.35, glide = msg[2] ? 1.8;
        if (drone.isNil) {
            drone = Synth(\jamDrone, [\freq, root, \amp, amp, \glide, glide]);
            "drone demarre (%.1f Hz)".format(root).postln;
        } {
            drone.set(\amp, amp, \glide, glide);
        };
    }, '/jam/drone');

    // /jam/key <tonic> <mode> <root_hz> <confidence>
    OSCdef(\jamKey, { |msg|
        var tonic = msg[1], mode = msg[2], hz = msg[3], conf = msg[4] ? 0.0;
        if (drone.notNil) {
            // Le mode colore le timbre plutot que la hauteur : mineur = plus
            // sombre, majeur = filtre plus ouvert.
            drone.set(\cutoff, if (mode == \min) { 700 } { 1100 });
            if (chordDriven.not) { root = hz; drone.set(\freq, root); };
        };
        "key -> % % (%.0f Hz, conf %.2f)".format(tonic, mode, hz, conf).postln;
    }, '/jam/key');

    // /jam/chord <root> <quality> <root_hz> <decision>
    //
    // C'est le message qui fait bouger l'harmonie : l'appli ne subit plus la
    // tonalite detectee, elle deroule une grille et annonce ou elle va.
    // `decision` dit qui a decide — "lead" l'appli, "follow" le joueur,
    // "anchor" un changement de tonalite qui a reancre la grille.
    OSCdef(\jamChord, { |msg|
        var quality = (msg[2] ? \min).asSymbol, hz = msg[3] ? 110.0;
        var decision = msg[4] ? \lead;
        var intervals = switch(quality, \min, { [3, 7] }, \sus4, { [5, 7] }, { [4, 7] });
        root = hz;
        chordDriven = true;
        if (drone.notNil) {
            drone.set(
                \freq, hz,
                \third, hz * intervals[0].midiratio,
                \fifth, hz * intervals[1].midiratio,
                \chordAmp, 1
            );
        };
        "chord -> % % (%.0f Hz, %)".format(msg[1], quality, hz, decision).postln;
    }, '/jam/chord');

    // /jam/energy <level> <density> : le niveau de jeu ouvre le filtre du drone.
    OSCdef(\jamEnergy, { |msg|
        var level = msg[1] ? 0.0;
        if (drone.notNil) {
            drone.set(\cutoff, 600 + (level * 1600));
        };
    }, '/jam/energy');

    // /jam/onset <strength>
    OSCdef(\jamOnset, { |msg|
        var strength = (msg[1] ? 1.0).clip(0.5, 4.0);
        Synth(\jamSpark, [\freq, root, \velocity, strength / 4.0]);
    }, '/jam/onset');

    // /jam/kick <velocity> <step>
    OSCdef(\jamKick, { |msg|
        Synth(\jamKick, [\velocity, msg[1] ? 0.8, \freq, root / 2]);
    }, '/jam/kick');

    // /jam/perc <velocity> <step>
    OSCdef(\jamPerc, { |msg|
        var velocity = msg[1] ? 0.6, step = msg[2] ? 0;
        Synth(\jamPerc, [
            \velocity, velocity,
            // Le pas de la grille pilote le panoramique : la sequence s'ouvre
            // en stereo au lieu de rester au centre.
            \pan, ((step % 8) / 7.0 * 1.2) - 0.6,
            \tone, 2400 + (step % 4 * 600)
        ]);
    }, '/jam/perc');

    // /jam/stop : extinction propre, appelee quand le process Python quitte.
    OSCdef(\jamStop, { |msg|
        if (drone.notNil) { drone.set(\gate, 0); drone = nil; };
        chordDriven = false;
        "jam arrete.".postln;
    }, '/jam/stop');

    "jam.scd pret — en ecoute sur le port % (prefixe /jam)".format(
        NetAddr.langPort
    ).postln;
});

// Filet de securite pendant les reglages : Ctrl-. coupe tout et libere les OSCdef.
CmdPeriod.doOnce({ OSCdef.freeAll; "OSCdef liberes.".postln; });
