/*
Copyright (c) 2010-2011, Advanced Micro Devices, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer. 

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution. 

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If you use the software (in whole or in part), you shall adhere to all applicable U.S., European, and other export
laws, including but not limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730 through
774), and E.U. Council Regulation (EC) No 1334/2000 of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR,
you hereby certify that, except pursuant to a license granted by the United States Department of Commerce Bureau of 
Industry and Security or as otherwise permitted pursuant to a License Exception under the U.S. Export Administration 
Regulations ("EAR"), you will not (1) export, re-export or release to a national of a country in Country Groups D:1,
E:1 or E:2 any restricted technology, software, or source code you receive hereunder, or (2) export to Country Groups
D:1, E:1 or E:2 the direct product of such technology or software, if such foreign produced direct product is subject
to national security controls as identified on the Commerce Control List (currently found in Supplement 1 to Part 774
of EAR).  For the most current Country Group listings, or for additional information about the EAR or your obligations
under those regulations, please refer to the U.S. Bureau of Industry and Security's website at http://www.bis.doc.gov/. 

*/
package com.amd.aparapi;

import java.util.List;

import com.amd.aparapi.device.Device;
import com.amd.aparapi.internal.kernel.KernelArg;
import com.amd.aparapi.internal.kernel.KernelManager;
import com.amd.aparapi.internal.kernel.KernelRunner;
import com.amd.aparapi.internal.kernel.KernelRunnerModular;
import com.amd.aparapi.internal.util.Reflection;

/**
 * This class is just adding modularity for kernel execution
 * Please see {@link #Kernel} for more information
 * 
 * <p>
 *
 * @author gfrost AMD Javalabs
 * @modifiedby Qunaibit
 * @version Alpha, May 15, 2016
 * 
 */
public abstract class KernelModular extends Kernel {


	public abstract class Entry {
		public abstract void run();

		public KernelModular execute(Range _range) {
			return (KernelModular.this.executePrepare("foo", _range, 1));
		}
	}

	private KernelRunnerModular kernelRunner = null;

	private boolean autoCleanUpArrays = false;

	private KernelState kernelState = new KernelState();

	/**
	 * Invoking this method flags that once the current pass is complete execution should be abandoned. Due to the complexity of intercommunication between java
	 * (or C) and executing OpenCL, this is the best we can do for general cancellation of execution at present. OpenCL 2.0 should introduce pipe mechanisms
	 * which will support mid-pass cancellation easily.
	 *
	 * <p>
	 * Note that in the case of thread-pool/pure java execution we could do better already, using Thread.interrupt() (and/or other means) to abandon execution
	 * mid-pass. However at present this is not attempted.
	 *
	 * @see #execute(int, int)
	 * @see #execute(Range, int)
	 * @see #execute(String, Range, int)
	 */
	public void cancelMultiPass() {
		if (kernelRunner == null) {
			return;
		}
		kernelRunner.cancelMultiPass();
	}

	public int getCancelState() {
		return kernelRunner == null ? KernelRunner.CANCEL_STATUS_FALSE : kernelRunner.getCancelState();
	}

	/**
	 * @see KernelRunner#getCurrentPass()
	 */
	public int getCurrentPass() {
		if (kernelRunner == null) {
			return KernelRunner.PASS_ID_COMPLETED_EXECUTION;
		}
		return kernelRunner.getCurrentPass();
	}

	/**
	 * @see KernelRunner#isExecuting()
	 */
	public boolean isExecuting() {
		if (kernelRunner == null) {
			return false;
		}
		return kernelRunner.isExecuting();
	}

	/**
	 * When using a Java Thread Pool Aparapi uses clone to copy the initial instance to each thread.
	 * 
	 * <p>
	 * If you choose to override <code>clone()</code> you are responsible for delegating to <code>super.clone();</code>
	 */
	@Override
	public KernelModular clone() {
		final KernelModular worker = (KernelModular) super.clone();

		// We need to be careful to also clone the KernelState
		worker.kernelState = worker.new KernelState(kernelState); // Qualified copy constructor

		worker.kernelState.setGroupIds(new int[] { 0, 0, 0 });

		worker.kernelState.setLocalIds(new int[] { 0, 0, 0 });

		worker.kernelState.setGlobalIds(new int[] { 0, 0, 0 });

		return worker;
	}

	public KernelState getKernelState() {
		return kernelState;
	}

	private KernelRunnerModular prepareKernelRunner() {
		if (kernelRunner == null) {
			kernelRunner = new KernelRunnerModular(this);
		}
		return kernelRunner;
	}

	/**
	 * Prepare execution of <code>_range</code> kernels.
	 * <p>
	 * When <code>kernel.execute(globalSize)</code> is invoked, Aparapi will schedule the execution of <code>globalSize</code> kernels. If the execution mode is
	 * GPU then the kernels will execute as OpenCL code on the GPU device. Otherwise, if the mode is JTP, the kernels will execute as a pool of Java threads on
	 * the CPU.
	 * <p>
	 * 
	 * @param _range
	 *            The number of Kernels that we would like to initiate.
	 * @returnThe Kernel instance (this) so we can chain calls to put(arr).execute(range).get(arr)
	 * 
	 */
	public synchronized KernelModular executePrepare(Range _range) {
		return (executePrepare(_range, 1));
	}

	@Override
	public String toString() {
		List<Device> preferredDevices = KernelManager.instance().getPreferences(this).getPreferredDevices(this);
		StringBuilder preferredDevicesSummary = new StringBuilder("{");
		for (int i = 0; i < preferredDevices.size(); ++i) {
			Device device = preferredDevices.get(i);
			preferredDevicesSummary.append(device.getShortDescription());
			if (i < preferredDevices.size() - 1) {
				preferredDevicesSummary.append("|");
			}
		}
		preferredDevicesSummary.append("}");
		return Reflection.getSimpleName(getClass()) + ", devices=" + preferredDevicesSummary.toString();
	}

	/**
	 * Prepare execution of <code>_range</code> kernels.
	 * <p>
	 * When <code>kernel.execute(_range)</code> is 1invoked, Aparapi will schedule the execution of <code>_range</code> kernels. If the execution mode is GPU
	 * then the kernels will execute as OpenCL code on the GPU device. Otherwise, if the mode is JTP, the kernels will execute as a pool of Java threads on the
	 * CPU.
	 * <p>
	 * Since adding the new <code>Range class</code> this method offers backward compatibility and merely defers to
	 * <code> return (execute(Range.create(_range), 1));</code>.
	 * 
	 * @param _range
	 *            The number of Kernels that we would like to initiate.
	 * @returnThe Kernel instance (this) so we can chain calls to put(arr).execute(range).get(arr)
	 * 
	 */
	public synchronized KernelModular executePrepare(int _range) {
		return (executePrepare(createRange(_range), 1));
	}

	protected Range createRange(int _range) {
		return Range.create(null, _range);
	}

	/**
	 * Prepare execution of <code>_passes</code> iterations of <code>_range</code> kernels.
	 * <p>
	 * When <code>kernel.execute(_range, _passes)</code> is invoked, Aparapi will schedule the execution of <code>_reange</code> kernels. If the execution mode
	 * is GPU then the kernels will execute as OpenCL code on the GPU device. Otherwise, if the mode is JTP, the kernels will execute as a pool of Java threads
	 * on the CPU.
	 * <p>
	 * 
	 * @param _passes
	 *            The number of passes to make
	 * @return The Kernel instance (this) so we can chain calls to put(arr).execute(range).get(arr)
	 * 
	 */
	public synchronized KernelModular executePrepare(Range _range, int _passes) {
		return (executePrepare("run", _range, _passes));
	}

	/**
	 * Prepare execution of <code>_passes</code> iterations over the <code>_range</code> of kernels.
	 * <p>
	 * When <code>kernel.execute(_range)</code> is invoked, Aparapi will schedule the execution of <code>_range</code> kernels. If the execution mode is GPU
	 * then the kernels will execute as OpenCL code on the GPU device. Otherwise, if the mode is JTP, the kernels will execute as a pool of Java threads on the
	 * CPU.
	 * <p>
	 * Since adding the new <code>Range class</code> this method offers backward compatibility and merely defers to
	 * <code> return (execute(Range.create(_range), 1));</code>.
	 * 
	 * @param _range
	 *            The number of Kernels that we would like to initiate.
	 * @returnThe Kernel instance (this) so we can chain calls to put(arr).execute(range).get(arr)
	 * 
	 */
	public synchronized KernelModular executePrepare(int _range, int _passes) {
		return (executePrepare(createRange(_range), _passes));
	}

	/**
	 * Prepare execution of <code>globalSize</code> kernels for the given entrypoint.
	 * <p>
	 * When <code>kernel.execute("entrypoint", globalSize)</code> is invoked, Aparapi will schedule the execution of <code>globalSize</code> kernels. If the
	 * execution mode is GPU then the kernels will execute as OpenCL code on the GPU device. Otherwise, if the mode is JTP, the kernels will execute as a pool
	 * of Java threads on the CPU.
	 * <p>
	 * 
	 * @param _entrypoint
	 *            is the name of the method we wish to use as the entrypoint to the kernel
	 * @return The Kernel instance (this) so we can chain calls to put(arr).execute(range).get(arr)
	 * 
	 */
	public synchronized KernelModular executePrepare(String _entrypoint, Range _range) {
		return (executePrepare(_entrypoint, _range, 1));
	}

	/**
	 * Prepare execution of <code>globalSize</code> kernels for the given entrypoint.
	 * <p>
	 * When <code>kernel.execute("entrypoint", globalSize)</code> is invoked, Aparapi will schedule the execution of <code>globalSize</code> kernels. If the
	 * execution mode is GPU then the kernels will execute as OpenCL code on the GPU device. Otherwise, if the mode is JTP, the kernels will execute as a pool
	 * of Java threads on the CPU.
	 * <p>
	 * 
	 * @param _entrypoint
	 *            is the name of the method we wish to use as the entrypoint to the kernel
	 * @return The Kernel instance (this) so we can chain calls to put(arr).execute(range).get(arr)
	 * 
	 */
	public synchronized KernelModular executePrepare(String _entrypoint, Range _range, int _passes) {
		return prepareKernelRunner().executePrepare(_entrypoint, _range, _passes);
	}

	/**
	 * Perform the kernel execution
	 * 
	 * @return kernel
	 */

	public synchronized KernelModular executeReady(int _range) {
		assert prepareKernelRunner().isExecuting() : "executePrepare(...) Should be called first!";
		return prepareKernelRunner().executeReady(createRange(_range));
	}
	
	public synchronized KernelModular executeReady() {
		assert prepareKernelRunner().isExecuting() : "executePrepare(...) Should be called first!";
		return prepareKernelRunner().executeReady();
	}


	public boolean isAutoCleanUpArrays() {
		return autoCleanUpArrays;
	}

	/**
	 * Property which if true enables automatic calling of {@link #cleanUpArrays()} following each execution.
	 */
	public void setAutoCleanUpArrays(boolean autoCleanUpArrays) {
		this.autoCleanUpArrays = autoCleanUpArrays;
	}

	/**
	 * Frees the bulk of the resources used by this kernel, by setting array sizes in non-primitive {@link KernelArg}s to 1 (0 size is prohibited) and invoking
	 * kernel execution on a zero size range. Unlike {@link #dispose()}, this does not prohibit further invocations of this kernel, as sundry resources such as
	 * OpenCL queues are <b>not</b> freed by this method.
	 *
	 * <p>
	 * This allows a "dormant" Kernel to remain in existence without undue strain on GPU resources, which may be strongly preferable to disposing a Kernel and
	 * recreating another one later, as creation/use of a new Kernel (specifically creation of its associated OpenCL context) is expensive.
	 * </p>
	 *
	 * <p>
	 * Note that where the underlying array field is declared final, for obvious reasons it is not resized to zero.
	 * </p>
	 */
	public synchronized void cleanUpArrays() {
		if (kernelRunner != null) {
			kernelRunner.cleanUpArrays();
		}
	}

	/**
	 * Release any resources associated with this Kernel.
	 * <p>
	 * When the execution mode is <code>CPU</code> or <code>GPU</code>, Aparapi stores some OpenCL resources in a data structure associated with the kernel
	 * instance. The <code>dispose()</code> method must be called to release these resources.
	 * <p>
	 * If <code>execute(int _globalSize)</code> is called after <code>dispose()</code> is called the results are undefined.
	 */
	public synchronized void dispose() {
		if (kernelRunner != null) {
			kernelRunner.dispose();
			kernelRunner = null;
		}
	}
}
