import { useId, useState } from 'react';

import { Icons } from '../icons.jsx';

// Accessible info tooltip: an info affordance that reveals help text on hover
// and on keyboard focus. The text is linked via aria-describedby so screen
// readers announce it. Used to explain options, thresholds, and impacts.
function Tooltip({ text, label = 'More information' }) {
  const [open, setOpen] = useState(false);
  const tooltipId = useId();

  if (!text) {
    return null;
  }

  return (
    <span className="tooltip-wrap">
      <button
        type="button"
        className="tooltip-trigger"
        aria-label={label}
        aria-describedby={open ? tooltipId : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        <Icons.info size={14} />
      </button>
      <span
        role="tooltip"
        id={tooltipId}
        className={`tooltip-bubble ${open ? 'open' : ''}`}
      >
        {text}
      </span>
    </span>
  );
}

export default Tooltip;
