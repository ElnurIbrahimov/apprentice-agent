package com.aura.ui.viewmodel

import android.app.Application
import com.aura.profile.UserProfile
import com.aura.profile.UserProfileStore
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals

/**
 * Tests for [ProfileViewModel] name/traits/facts state machine.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ProfileViewModelTest {

    private val store = mockk<UserProfileStore>(relaxed = true)
    private val profileFlow = MutableStateFlow(UserProfile())

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        every { store.profile } returns profileFlow
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial state reflects store profile`() = runTest {
        profileFlow.value = UserProfile(name = "Elnur", traits = listOf("technical"), facts = listOf("Lives in Baku"))
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        assertEquals("Elnur", vm.state.value.name)
        assertEquals(listOf("technical"), vm.state.value.traits)
        assertEquals(listOf("Lives in Baku"), vm.state.value.facts)
    }

    @Test
    fun `setName stores trimmed name`() = runTest {
        coEvery { store.update(name = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.setName("  Elnur  ")
        coVerify { store.update(name = "Elnur") }
    }

    @Test
    fun `setName clears name when blank`() = runTest {
        coEvery { store.update(name = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.setName("   ")
        coVerify(exactly = 1) { store.update(name = null) }
    }

    @Test
    fun `addTrait appends non-duplicate`() = runTest {
        profileFlow.value = UserProfile(traits = listOf("technical"))
        coEvery { store.update(traits = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.addTrait("concise")
        coVerify { store.update(traits = listOf("technical", "concise")) }
    }

    @Test
    fun `addTrait ignores blank and duplicate`() = runTest {
        profileFlow.value = UserProfile(traits = listOf("technical"))
        coEvery { store.update(traits = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.addTrait("  ")
        vm.addTrait("technical")
        coVerify(exactly = 0) { store.update(traits = any()) }
    }

    @Test
    fun `removeTrait filters out trait`() = runTest {
        profileFlow.value = UserProfile(traits = listOf("technical", "concise"))
        coEvery { store.update(traits = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.removeTrait("technical")
        coVerify { store.update(traits = listOf("concise")) }
    }

    @Test
    fun `addFact appends non-duplicate`() = runTest {
        profileFlow.value = UserProfile(facts = listOf("Lives in Baku"))
        coEvery { store.update(facts = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.addFact("Works remotely")
        coVerify { store.update(facts = listOf("Lives in Baku", "Works remotely")) }
    }

    @Test
    fun `removeFact filters out fact`() = runTest {
        profileFlow.value = UserProfile(facts = listOf("Lives in Baku", "Works remotely"))
        coEvery { store.update(facts = any()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.removeFact("Lives in Baku")
        coVerify { store.update(facts = listOf("Works remotely")) }
    }

    @Test
    fun `clear resets name traits and facts`() = runTest {
        coEvery { store.update(name = "", traits = emptyList<String>(), facts = emptyList<String>()) } returns Unit
        val vm = ProfileViewModel(mockk(relaxed = true), store)
        vm.clear()
        coVerify { store.update(name = "", traits = emptyList<String>(), facts = emptyList<String>()) }
    }
}
