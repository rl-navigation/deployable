from __future__ import print_function, division
import collections, sys

HYPERS = collections.OrderedDict(); DEFTS = {}; HINTS = {}

def arg(tag, default=None, hint=None):
    if default is not None: HYPERS[tag] = type(default)((sys.argv[sys.argv.index(tag)+1])) if tag in sys.argv else default
    else                  : HYPERS[tag] = tag in sys.argv
    DEFTS[tag] = default
    HINTS[tag] = hint
    exec("{}={}".format(tag.replace("--","").replace("-","").upper(),repr(HYPERS[tag])),globals())

def argvalidate():
    if "--help" in sys.argv or "-h" in sys.argv or len(sys.argv)==1:
        al = max([len(k)      for k in HYPERS.keys()])
        hl = max([len(str(v)) for v in HINTS .values()])
        opts = []
        for k,v in HYPERS.items():
            tag  = "{:{l}}".format(k, l=al)
            name = "{:{l}}".format(k.replace("--","").upper() if type(v) is not bool else "", l=al)
            hint = "{:{l}}".format(HINTS[k] if HINTS[k] is not None else "", l=hl)
            deft = "[default={}]".format(DEFTS[k]) if DEFTS[k] is not None else ""
            opts.append(" ".join([tag, name, hint, deft]))
        opts = "\n".join(opts)
        print("Usage:\n{} [options]\n\nOptions:\n{}".format(sys.argv[0], opts))
        sys.exit(0)
    elif any([a not in HYPERS.keys() for a in sys.argv[1:] if a.startswith("--")]):
        print("Unrecognized arguments {}".format([a for a in sys.argv[1:] if a not in HYPERS.keys() and a.startswith("-")]))
        sys.exit(1)

