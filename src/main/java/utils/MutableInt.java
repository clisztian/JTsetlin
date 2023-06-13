package utils;

public class MutableInt implements Comparable<MutableInt> {
	  int value = 1; // note that we start at 1 since we're counting
	  public void increment () { ++value;      }
	  public int  get ()       { return value; }
	  public void increment (int inc) { value += inc; }

	public MutableInt(int value) {
		this.value = value;
	}
	public MutableInt() {
		this.value = 1;
	}

	@Override
	public int compareTo(MutableInt o) {
		return Integer.compare(get(), o.get());
	}
}